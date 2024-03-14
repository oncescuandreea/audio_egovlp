"""Module for computing performance metrics

"""
import os
import pickle
import random
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F

from utils import calculate_from_split, mAP, nDCG, print_mAP, print_nDCG


def t2v_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [
        [
            np.ravel_multi_index([ii, jj], (num_queries, num_vids))
            for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)
        ]
        for jj in range(num_vids)
    ]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin,
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    break_ties = "optimistically"
    # break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            if False:
                print("Running slower code to verify rank averaging across ties")
                # slow, but more interpretable version, used for testing
                avg_cols_slow = [
                    np.mean(cols[rows == idx]) for idx in range(num_queries)
                ]
                assert np.array_equal(
                    avg_cols, avg_cols_slow
                ), "slow vs fast difference"
                print("passed num check")
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb

        ipdb.set_trace()
    assert cols.size == num_queries, msg

    if False:
        # overload mask to check that we can recover the scores for single-query
        # retrieval
        print("DEBUGGING MODE")
        query_masks = np.zeros_like(query_masks)
        query_masks[:, 0] = 1  # recover single query score

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    if False:
        # sanity check against old logic for square matrices
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        _, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        assert np.array_equal(cols_old, cols), "new metric doesn't match"

    return cols2metrics(cols, num_queries)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    # switch axes of text and video
    sims = sims.T

    if False:
        # experiment with toy example
        sims = np.ones((3, 3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2, :] = 2
        query_masks = None

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1e8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            # Set missing queries to have a distance of infinity.  A missing query
            # refers to a query position `n` for a video that had less than `n`
            # captions (for example, a few MSRVTT videos only have 19 queries)
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                # skip rankings of missing captions
                continue
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        sorted_dists = np.sort(dists, axis=1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index=True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg

        # visualise the distance matrix
        import sys

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig  # NOQA

        plt.matshow(dists)
        zs_dispFig()

    return cols2metrics(query_ranks, num_queries)


def cols2metrics(cols, num_queries, query_masks_class=None):
    if query_masks_class is not None:
        new_query_number = int(num_queries) - np.count_nonzero(query_masks_class == 0)
        new_cols = np.zeros(new_query_number)
        counter = 0
        query_masks_class = query_masks_class.reshape(cols.shape)
        for loc, query_mask in enumerate(query_masks_class):
            if query_mask == 1:
                new_cols[counter] = cols[loc]
                counter += 1
        cols = new_cols
        num_queries = new_query_number
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    print(f"Results calculated for {num_queries} queries")
    return metrics


def egomcq_accuracy_metrics(preds, labels, types, group_list=None):
    metrics = {}
    type_list = torch.unique(types)
    # group_list = ["Intra-video", "Inter-video"]
    group_list = (
        group_list if group_list is not None else ["Inter-video", "Intra-video"]
    )
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct / total
        # metrics['accuracy_p' + str(type_i.item())] = accuracy * 100
        metrics[group_i] = accuracy * 100
    return metrics


def initialise_nDCG_values(relevancy_matrix):
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {"v": vis_k_counts, "t": txt_k_counts}
    IDCG_dict = {"v": vis_IDCG, "t": txt_IDCG}

    return IDCG_dict, k_counts_dict


def initialise_jpose_nDCG_values(relevancy_matrix):
    action_IDCG, action_k_values = initialise_nDCG_values(relevancy_matrix)

    dataset = {}
    dataset["action"] = {}
    dataset["action"]["IDCG"] = action_IDCG
    dataset["action"]["k_values"] = action_k_values
    return dataset


def mir_metrics(similarity_matrix, idx_arr):
    # considered unique narrations for evaluation of EPIC
    metrics = {}

    path_dataframes = "dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations"
    video_id = pd.read_csv(
        os.path.join(path_dataframes, "EPIC_100_retrieval_test.csv")
    ).values[:, 0]
    text_id = pd.read_csv(
        os.path.join(path_dataframes, "EPIC_100_retrieval_test_sentence.csv")
    ).values[:, 0]
    similarity_matrix = (similarity_matrix + 1) / 2

    indexes = []
    for elem in text_id:
        try:
            indexes.append(video_id.tolist().index(elem))
        except:
            print(f"error happened when index of {elem}.")

    order = []
    idx_arr_list = idx_arr.tolist()
    for i in range(len(video_id)):
        order.append(idx_arr_list.index(i))
    similarity_matrix = similarity_matrix[order, :][:, order]

    similarity_matrix = similarity_matrix.T[:, indexes]

    path_relevancy = "dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"
    pkl_file = open(path_relevancy, "rb")
    relevancy = pickle.load(pkl_file)

    dataset = initialise_jpose_nDCG_values(relevancy)
    vis_nDCG = nDCG.calculate_nDCG(
        similarity_matrix,
        relevancy,
        dataset["action"]["k_values"]["v"],
        IDCG=dataset["action"]["IDCG"]["v"],
    )
    txt_nDCG = nDCG.calculate_nDCG(
        similarity_matrix.T,
        relevancy.T,
        dataset["action"]["k_values"]["t"],
        IDCG=dataset["action"]["IDCG"]["t"],
    )
    metrics["nDCG_V2T"] = vis_nDCG * 100
    metrics["nDCG_T2V"] = txt_nDCG * 100
    metrics["nDCG_AVG"] = 100 * (vis_nDCG + txt_nDCG) / 2

    vis_mAP = mAP.calculate_mAP(similarity_matrix, relevancy)
    txt_mAP = mAP.calculate_mAP(similarity_matrix.T, relevancy.T)
    metrics["mAP_V2T"] = vis_mAP * 100
    metrics["mAP_T2V"] = txt_mAP * 100
    metrics["mAP_AVG"] = 100 * (vis_mAP + txt_mAP) / 2
    return metrics


def map(submission_array, gt_array):
    """Returns mAP, weighted mAP, and AP array"""
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float("nan"))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float)
    return m_ap, w_ap, m_aps


def charades_metrics(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    metrics = {}
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    m_ap, w_ap, m_aps = map(fix, gt_array)
    metrics["mAP"] = m_ap
    # metrics['wAP'] = w_ap
    # metrics['mAPs'] = m_aps
    return metrics


def oscc_metrics(preds, labels):
    metrics = {}
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct / total
    metrics["accuracy"] = accuracy * 100
    return metrics


def pnr_metrics(
    preds,
    labels,
    sc_labels,
    fps,
    parent_start_frames,
    parent_end_frames,
    parent_pnr_frames,
):
    metrics = {}
    distance_list = list()
    for (
        pred,
        label,
        sc_label,
        parent_start_frame,
        parent_end_frame,
        parent_pnr_frame,
        ind_fps,
    ) in zip(
        preds,
        labels,
        sc_labels,
        parent_start_frames,
        parent_end_frames,
        parent_pnr_frames,
        fps,
    ):
        # import pdb; pdb.set_trace()
        if sc_label.item() == 1:
            keyframe_loc_pred = torch.argmax(pred).item()
            # import pdb; pdb.set_trace()
            keyframe_loc_pred_mapped = (
                (parent_end_frame - parent_start_frame) / 16 * keyframe_loc_pred
            )
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = parent_pnr_frame.item() - parent_start_frame.item()
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame / ind_fps.item()
            # print(keyframe_loc_pred_mapped, gt,  err_frame, err_sec)
            distance_list.append(err_sec)
    if len(distance_list) == 0:
        # If evaluating the trained model, use this
        # Otherwise, Lightning expects us to give a number.
        # Due to this, the Tensorboard graphs' results for keyframe distance
        # will be a little inaccurate.
        metrics["keyframe_distance"] = np.mean(0.0)
    metrics["keyframe_distance"] = np.mean(distance_list)
    # import pdb;
    # pdb.set_trace()
    return metrics


def sim_matrix(a, b, eps=1e-8, norm: str = ""):
    """
    added eps for numerical stability
    """
    if a.ndim == 3 == b.ndim:
        a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    else:
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    if a_norm.ndim == 3 == b_norm.ndim:
        sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
        sim_mt = sim_mt.squeeze(1)
    else:
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    if norm != "":
        if norm == "max":
            sim_mt = F.normalize(sim_mt, dim=1)
        elif norm == "one":
            sim_mt = (sim_mt.T / (sim_mt.sum(dim=1))).T
    return sim_mt.detach()


def sim_norm(sim_mt: np.array, norm: str = ""):
    if norm == "":
        sim_mt_t = sim_mt.T
    elif norm == "max":
        sim_mt_t = (sim_mt / np.linalg.norm(sim_mt.T, ord=2, axis=1)).T
        sim_mt = (sim_mt.T / np.linalg.norm(sim_mt, ord=2, axis=1)).T
    elif norm == "one":
        sim_mt_t = (sim_mt / sim_mt.T.sum(dim=1)).T
        sim_mt = (sim_mt.T / sim_mt.sum(dim=1)).T
    return sim_mt, sim_mt_t


def sim_matrices_vid_only(text_embeds, vid_embeds, indexes):
    """
    This function only calculates similarity matrix for video.
    The other similarity matrices are either repeated or are checking to make sure that
    normalisation of the matrices results in same metrics
    """
    similarity_matrix = (
        sim_matrix(text_embeds, vid_embeds).numpy() + 1
    ) / 2  # no_texts x no_videos
    similarity_matrix_tot = similarity_matrix.copy()
    with open(
        "/scratch/shared/beegfs/oncescu/data/epic-kitchens-100-annotations/retrieval_annotations/similarity_matrix_for_vid_gpt_epicsounds.npy",
        "wb",
    ) as f:
        np.save(f, similarity_matrix_tot)
    similarity_matrix = similarity_matrix.T[:, indexes]  # no_vid x no_unique_texts
    # sim matrix 2 is making sure that regardless of the normalisation applied to the matrix, the metrics stay the same

    similarity_matrix2, similarity_matrix2_t = sim_norm(similarity_matrix, "max")
    return (
        similarity_matrix,
        similarity_matrix2,
        similarity_matrix2_t,
        similarity_matrix2,
        similarity_matrix2.T,
        similarity_matrix_tot,
    )


class Version(Enum):
    V1 = "v1"
    V8 = "v8"
    V9 = "v9"
    V10 = "v10"

    @property
    def description(self):
        descriptions = {
            Version.V1: "original",  # calculate vid and aud sim matrices, average them, then transpose
            Version.V2: "nothing",
            Version.V3: "puts different weights on the matrices and could also normalise them. Since they are not squared, it takes care of this depending on the direction of the task V->T or T->V",
            Version.V4: "uses a masking approach for the audio content to decide if to completely/partially ignore it",
            Version.V5: "uses a mask of full 0s",
            Version.V6: "uses a mask of full 1s",
            Version.V7: "uses a mask of [no_audios] random 1s",
            Version.V8: "low_subset",  # calculates metrics only on low subset
            Version.V9: "moderate_subset",  # calculates metrics only on moderate subset
            Version.V10: "high_subset",  # calculates metrics only on high subset
        }
        return descriptions.get(self, "Unknown version")


def compute_v1_similarity_matrix(
    similarity_matrix_vid, similarity_matrix_aud, indexes, mask_array=None
):
    similarity_matrix = (
        (similarity_matrix_vid + similarity_matrix_aud * mask_array) / (mask_array + 1)
        + 1
    ) / 2
    similarity_matrix = similarity_matrix.T[:, indexes]
    similarity_matrix_t = similarity_matrix.T
    return similarity_matrix, similarity_matrix_t


def sim_matrices_both(
    text_embeds_aud: torch.tensor,
    text_embeds_vid: torch.tensor,
    vid_embeds: torch.tensor,
    aud_embeds: torch.tensor,
    indexes: list,
    config,
    relevancy,
    dataset,
    version: str = "v1",
    mask_array: np.array = None,
    mask_dict_aud_rel: dict = None,
    output: str = "",
):
    """
    Multiple versions of similarity matrices:
    v1 = calculate vid and aud sim matrices, average them, then transpose

    """
    version_to_name = {
        "v1": "original",
        "v8": "low_subset",
        "v9": "moderate_subset",
        "v10": "high_subset",
    }
    similarity_matrix_vid = sim_matrix(text_embeds_vid, vid_embeds).numpy()
    similarity_matrix_aud = sim_matrix(text_embeds_aud, aud_embeds).numpy()
    assert (
        1 <= int(version[1:]) <= 11
    ), "Version not supported for joint similarity matrix."
    version_to_level = {"v8": "low", "v9": "moderate", "v10": "high"}
    if version in set(["v1", "v2", "v3", "v4"]):
        if version == "v1":
            no_elems = similarity_matrix_vid.shape[0]
            mask_array = np.array([1.0] * no_elems)[None, :]
        elif version == "v4":
            assert (
                mask_array is not None if version == "v4" else True
            ), "Need to specify the mask array for this similarity matrix."
        else:
            mask_array_v7 = np.zeros_like(np.array(mask_array))
            no_used_auds = (np.array(mask_array) == 1.0).sum()
            no_elems = len(mask_array)
            rand_indexes = random.sample(range(0, no_elems), no_used_auds)

            mask_array_v7[rand_indexes] = 1
            mask_array = {
                "v5": np.array([0.0] * no_elems)[None, :],
                "v6": np.array([1.0] * no_elems)[None, :],
                "v7": mask_array_v7[None, :],
            }.get(version, np.array(mask_array)[None, :])

        similarity_matrix, similarity_matrix_t = compute_v1_similarity_matrix(
            similarity_matrix_vid, similarity_matrix_aud, indexes, mask_array
        )
    elif version in ["v8", "v9", "v10"]:
        level = version_to_level[version]
        level_mask_array = np.array(mask_dict_aud_rel[level])
        similarity_matrix = (
            (similarity_matrix_vid + similarity_matrix_aud * level_mask_array)
            / (level_mask_array + 1)
            + 1
        ) / 2
        similarity_matrix = similarity_matrix.T[:, indexes]
        similarity_matrix_t = similarity_matrix.T
    else:
        similarity_matrix_vid = similarity_matrix_vid.T[:, indexes]
        similarity_matrix_aud = similarity_matrix_aud.T[:, indexes]
        if version == "v2":
            similarity_matrix = (
                (similarity_matrix_vid + similarity_matrix_aud) / 2 + 1
            ) / 2
            similarity_matrix_t = similarity_matrix.T
        elif version == "v3":
            print(f'norm is {config.config["trainer"]["normalisation"]}')
            similarity_matrix_vid, similarity_matrix_vid_t = sim_norm(
                similarity_matrix_vid, config.config["trainer"]["normalisation"]
            )
            similarity_matrix_aud, similarity_matrix_aud_t = sim_norm(
                similarity_matrix_aud, config.config["trainer"]["normalisation"]
            )
            similarity_matrix = (
                (
                    config["trainer"]["vtoa_ratio"] * similarity_matrix_vid
                    + similarity_matrix_aud
                )
                / (config["trainer"]["vtoa_ratio"] + 1)
                + 1
            ) / 2
            similarity_matrix_t = (
                (
                    config["trainer"]["vtoa_ratio"] * similarity_matrix_vid_t
                    + similarity_matrix_aud_t
                )
                / (config["trainer"]["vtoa_ratio"] + 1)
                + 1
            ) / 2
        else:
            raise NotImplementedError(
                "Only commented code available, not implemented yet"
            )
            identity_mask = np.identity(len(similarity_matrix_vid))
            uncertain_vid = (
                ma.masked_array(similarity_matrix_vid, identity_mask).sum(axis=1) > 0.7
            ).astype(int)

            ####### try filtering visual only descriptions
            # mask_tensor = torch.tensor(mask_array).unsqueeze(0).T

            ####### done

            # similarity_matrix = uncertain_vid*similarity_matrix_vid + (1-uncertain_vid)* similarity_matrix_aud
    vis_nDCG = nDCG.calculate_nDCG(
        similarity_matrix,
        relevancy,
        dataset["action"]["k_values"]["v"],
        IDCG=dataset["action"]["IDCG"]["v"],
        reduction=None,
    )
    txt_nDCG = nDCG.calculate_nDCG(
        similarity_matrix_t,
        relevancy.T,
        dataset["action"]["k_values"]["t"],
        IDCG=dataset["action"]["IDCG"]["t"],
        reduction=None,
    )
    print(f"Version is {version_to_name[version]}")
    output += f"Version is {version_to_name[version]}"
    output = print_nDCG(vis_nDCG, txt_nDCG, output)

    vis_mAP = mAP.calculate_mAP(similarity_matrix, relevancy, reduction=None)
    txt_mAP = mAP.calculate_mAP(similarity_matrix_t, relevancy.T, reduction=None)

    output = print_mAP(vis_mAP, txt_mAP, output)

    if version == "v1":
        print("--------------------------")
        output += "--------------------------"
        output = calculate_from_split(
            output,
            mask_dict_aud_rel,
            config,
            indexes,
            vis_nDCG,
            txt_nDCG,
            vis_mAP,
            txt_mAP,
        )
        print("--------------------------")
        output += "--------------------------"
    return output


def sim_matrix_mm(a, b):
    print("Got to sim matrix mm line 92")
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt.detach().numpy()


def softmax_numpy(sim, dim=0):
    sim = torch.Tensor(sim)
    sim = F.softmax(sim, dim=dim)
    return sim.numpy()
