import data
from numba import njit, prange
import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split


def calculate_similarity_matrix(mat: np.array, method: str = 'cosine') -> np.array:
    """
    Calculates the pairwise similarity of vectors within a matrix.
    For matrix of shape (A, B), returns a matrix of shape (A, A) where each entry is the similarity between the vectors at the corresponding row indices.
    For matrix of shape (N, A, B), returns a matrix of shape (N, A, A).

    Parameters
    ----------
    mat : np.array
        Matrix of vectors to calculate similarity between.
    method : str, optional
        Method to use for calculating similarity. One of ['cosine', 'correlation'], by default 'cosine'.
    
    Returns
    -------
    np.array
        Matrix of pairwise similarities.
    """
    if method == 'cosine':
        if len(mat.shape)==2:
            return fast_cosine_matrix_2d(mat)
        else:
            return fast_cosine_matrix_3d(mat)
    elif method == 'correlation':
        if type(mat) is list or len(mat.shape)==3:
            corrs = np.zeros((len(mat),mat[0].shape[0],mat[0].shape[0]))
            for i in range(len(mat)):
                corrs[i] = np.corrcoef(mat[i])
            return corrs
        return np.corrcoef(mat)
    else:
        raise Exception(f'Unknown similarity method {method} - try one of ["cosine","correlation"].')


def filter_nan_similarity(reps: np.array, sim: np.array) -> tuple[list[np.array], np.array, np.array]:
    """
    Filters out contexts and objects that have no representation or similarity values.
    This is done to avoid NaN values in the similarity matrix, which can cause issues with some similarity methods.

    Parameters
    ----------
    reps : np.array
        Array of representations to filter.
    sim : np.array
        Array of similarities to filter.
    
    Returns
    -------
    tuple[list[np.array], np.array, np.array]
        Tuple of filtered representations, filtered similarity matrix, and the number of objects in each context.
    """
    contexts_to_keep = [context for context in range(len(sim)) if np.isnan(sim[context]).mean()<.5]
    objects_to_keep = np.zeros((len(contexts_to_keep),sim.shape[1]),dtype=int)
    num_objects_to_keep = np.zeros(len(contexts_to_keep),dtype=int)
    for i,context in enumerate(contexts_to_keep):
        non_null_objects = np.where(reps[context].max(axis=1)>0)[0]
        num_objects_to_keep[i] = len(non_null_objects)
        objects_to_keep[i,:len(non_null_objects)] = non_null_objects
    return contexts_to_keep, objects_to_keep, num_objects_to_keep


@njit(parallel=True,fastmath=True)
def fast_rsa_bootstrap(mat0: np.array, mat1: np.array, mat2: np.array, sample_idxs: list[np.array],
                       sample_idx_lengths: np.array, n_sims: int) -> tuple[np.array, np.array]:
    """
    Calculates the correlations between two similarity matrices and a reference similarity matrix,
    using a bootstrap procedure to estimate the null distribution.

    Parameters
    ----------
    mat0 : np.array
        Reference similarity matrix.
    mat1 : np.array
        First similarity matrix to compare to reference.
    mat2 : np.array
        Second similarity matrix to compare to reference.
    sample_idxs : list[np.array]
        List of arrays of indices to sample from for each context.
    sample_idx_lengths : np.array
        Array of lengths of each array in `sample_idxs`.
    n_sims : int
        Number of bootstrap simulations to run.
    
    Returns
    -------
    tuple[np.array, np.array]
        Tuple of arrays of correlation values for each bootstrap simulation.
    """
    n_iters = (mat0.shape[1]*(mat0.shape[1]-1))//2  # Number of pairwise comparisons
    n_contexts = len(sample_idxs)
    corrs_y = np.zeros((n_sims,n_contexts))
    corrs_z = np.zeros((n_sims,n_contexts))
    for sim in prange(n_sims):
        x, y, z = np.zeros((n_iters,)), np.zeros((n_iters,)), np.zeros((n_iters,))
        for context in range(n_contexts):
            n_objects = sample_idx_lengths[context]
            x_sum, y_sum, z_sum = 0, 0, 0
            counter = 0
            for i in range(n_objects):
                idx1 = np.random.choice(sample_idxs[context])
                for j in range(i+1,n_objects):
                    idx2 = np.random.choice(sample_idxs[context])
                    while idx2==idx1:
                        idx2 = np.random.choice(sample_idxs[context])
                    x[counter] = mat0[context, idx1, idx2]
                    y[counter] = mat1[context, idx1, idx2]
                    z[counter] = mat2[context, idx1, idx2]
                    x_sum += x[counter]
                    y_sum += y[counter]
                    z_sum += z[counter]
                    counter += 1
            x_sum /= counter
            y_sum /= counter
            z_sum /= counter
            xval, yval, zval = 0, 0, 0
            xy_corr_numerator = 0
            xz_corr_numerator = 0
            corrs_denomenator_x = 0
            corrs_denomenator_y = 0
            corrs_denomenator_z = 0
            for i in range(counter):
                xval = x[i]-x_sum
                yval = y[i]-y_sum
                zval = z[i]-z_sum
                xy_corr_numerator += xval*yval
                xz_corr_numerator += xval*zval
                corrs_denomenator_x += xval*xval
                corrs_denomenator_y += yval*yval
                corrs_denomenator_z += zval*zval
            corrs_y[sim,context] = xy_corr_numerator/np.sqrt(corrs_denomenator_x*corrs_denomenator_y)
            corrs_z[sim,context] = xz_corr_numerator/np.sqrt(corrs_denomenator_x*corrs_denomenator_z)
    return corrs_y, corrs_z


@njit(parallel=True,fastmath=True)
def fast_correlation_splithalf(vec1: np.array, mat1: np.array, vec2: np.array, mat2: np.array,
                               n_sims: np.array, replace_nan = np.nan) -> tuple[np.array, np.array]:
    """
    Calculates the correlation between two vectors and two matrices, using a split-half procedure to estimate the null distribution.

    Parameters
    ----------
    vec1 : np.array
        First vector to correlate.
    mat1 : np.array
        First matrix to correlate.
    vec2 : np.array
        Second vector to correlate.
    mat2 : np.array
        Second matrix to correlate.
    n_sims : int
        Number of split-half simulations to run.
    replace_nan : float, optional
        Value to replace NaN values with, by default np.nan.
    
    Returns
    -------
    tuple[np.array, np.array]
        Tuple of arrays of correlation values for each split-half simulation.
    """
    within_corrs, across_corrs = np.zeros((n_sims,)), np.zeros((n_sims,))
    for sim in prange(n_sims):
        num_idxs1 = len(vec1)
        num_idxs1 -= num_idxs1%2
        num_idxs2 = len(vec2)
        num_idxs2 -= num_idxs2%2
        idxs1 = np.arange(num_idxs1)
        idxs2 = np.arange(num_idxs2)
        np.random.shuffle(idxs1)
        np.random.shuffle(idxs2)
        vec1_half1, vec1_half2 = vec1[idxs1[:num_idxs1//2]], vec1[idxs1[num_idxs1//2:]]
        mat1_half1, mat1_half2 = mat1[idxs1[:num_idxs1//2]], mat1[idxs1[num_idxs1//2:]]
        vec2_half1, vec2_half2 = vec2[idxs2[:num_idxs2//2]], vec2[idxs2[num_idxs2//2:]]
        mat2_half1, mat2_half2 = mat2[idxs2[:num_idxs2//2]], mat2[idxs2[num_idxs2//2:]]

        corrs_1h1 = fast_correlation_vec_mat(vec1_half1,mat1_half1,replace_nan=replace_nan)
        corrs_1h2 = fast_correlation_vec_mat(vec1_half2,mat1_half2,replace_nan=replace_nan)
        corrs_2h1 = fast_correlation_vec_mat(vec2_half1,mat2_half1,replace_nan=replace_nan)
        corrs_2h2 = fast_correlation_vec_mat(vec2_half2,mat2_half2,replace_nan=replace_nan)

        within_corr = (fast_correlation_vec_vec(corrs_1h1,corrs_1h2,replace_nan=replace_nan)+
                       fast_correlation_vec_vec(corrs_2h1,corrs_2h2,replace_nan=replace_nan)
                       )/2
        across_corr = (fast_correlation_vec_vec(corrs_1h1,corrs_2h2,replace_nan=replace_nan)+
                       fast_correlation_vec_vec(corrs_2h1,corrs_1h2,replace_nan=replace_nan)
                       )/2
        within_corrs[sim] = within_corr
        across_corrs[sim] = across_corr
    return within_corrs, across_corrs


def cosine_splithalf(small1: np.array, large1: np.array, small2: np.array, large2: np.array,
                     n_sims: int = 10000) -> tuple[np.array, np.array]:
    """
    Calculates the cosine similarity between two sets of vectors,
    using a split-half procedure to estimate the null distribution.
    Used to measure the similarity of size representations between two categories.

    Parameters
    ----------
    small1 : np.array
        First set of small vectors.
    large1 : np.array
        First set of large vectors.
    small2 : np.array
        Second set of small vectors.
    large2 : np.array
        Second set of large vectors.
    n_sims : int, optional
        Number of split-half simulations to run, by default 10000.
    
    Returns
    -------
    tuple[np.array, np.array]
        Tuple of arrays of cosine similarity values within and across the two categories
        for each split-half simulation.
    """

    within_angles, across_angles = np.zeros((n_sims,)), np.zeros((n_sims,))
    for sim in range(n_sims):
        small1_train, small1_test = train_test_split(small1,test_size=.5)
        large1_train, large1_test = train_test_split(large1,test_size=.5)
        small2_train, small2_test = train_test_split(small2,test_size=.5)
        large2_train, large2_test = train_test_split(large2,test_size=.5)
        
        small1_to_large1_train = large1_train.mean(axis=0)-small1_train.mean(axis=0)
        small1_to_large1_test = large1_test.mean(axis=0)-small1_test.mean(axis=0)
        small2_to_large2_train = large2_train.mean(axis=0)-small2_train.mean(axis=0)
        small2_to_large2_test = large2_test.mean(axis=0)-small2_test.mean(axis=0)
        
        within_angles[sim] = (cosine(small1_to_large1_train,small1_to_large1_test)+
                              cosine(small2_to_large2_train,small2_to_large2_test))/2
        across_angles[sim] = (cosine(small1_to_large1_train,small2_to_large2_test)+
                              cosine(small2_to_large2_train,small1_to_large1_test))/2
    return within_angles,across_angles


def cosine_splithalf_paired(small1: np.array, large1: np.array, small2: np.array, large2: np.array,
                            small1_complement: np.array, large1_complement: np.array, small2_complement: np.array,
                            large2_complement: np.array, n_sims: int = 10000) -> tuple[np.array, np.array]:
    """
    Calculates the cosine similarity between two sets of vectors paired by category,
    using a split-half procedure to estimate the null distribution.

    Used to measure the similarity of size representations between two categories in the comparison models that are
    trained on size-complement data. For example, the comparison network is trained on small animals like hamsters
    and size-complement animals like large hamsters. The `small1` array contains representations of the small animals
    (like hamsters), while the `large1` array contains representations of the size-complement animals
    (like large hamsters).

    Parameters
    ----------
    small1 : np.array
        First set of small vectors.
    large1 : np.array
        First set of large vectors.
    small2 : np.array
        Second set of small vectors.
    large2 : np.array
        Second set of large vectors.
    small1_complement : np.array
        Complement of first set of small vectors.
    large1_complement : np.array
        Complement of first set of large vectors.
    small2_complement : np.array
        Complement of second set of small vectors.
    large2_complement : np.array
        Complement of second set of large vectors.
    n_sims : int, optional
        Number of split-half simulations to run, by default 10000.
    
    Returns
    -------
    tuple[np.array, np.array]
        Tuple of arrays of cosine similarity values within and across the two categories
        for each split-half simulation.
    """
    within_angles, across_angles = np.zeros((n_sims,)), np.zeros((n_sims,))
    for sim in range(n_sims):
        small1_train, small1_test, small1b_train, small1b_test = train_test_split(small1,small1_complement,test_size=.5)
        large1_train, large1_test, large1b_train, large1b_test = train_test_split(large1,large1_complement,test_size=.5)
        small2_train, small2_test, small2b_train, small2b_test = train_test_split(small2,small2_complement,test_size=.5)
        large2_train, large2_test, large2b_train, large2b_test = train_test_split(large2,large2_complement,test_size=.5)
    
        small1_to_large1_train = ((small1b_train-small1_train).mean(axis=0)+(large1_train-large1b_train).mean(axis=0))/2
        small2_to_large2_train = ((small2b_train-small2_train).mean(axis=0)+(large2_train-large2b_train).mean(axis=0))/2
        small1_to_large1_test = ((small1b_test-small1_test).mean(axis=0)+(large1_test-large1b_test).mean(axis=0))/2
        small2_to_large2_test = ((small2b_test-small2_test).mean(axis=0)+(large2_test-large2b_test).mean(axis=0))/2
    
        within_angles[sim] = (cosine(small1_to_large1_train,small1_to_large1_test)+
                        cosine(small2_to_large2_train,small2_to_large2_test))/2
        across_angles[sim] = (cosine(small1_to_large1_train,small2_to_large2_test)+
                        cosine(small2_to_large2_train,small1_to_large1_test))/2
    return within_angles,across_angles


def calculate_task_similarity_matrix(feature_sim: np.array, contexts_to_keep: list[np.array],
                                     objects_to_keep: np.array, num_objects_to_keep: np.array) -> np.array:
    """
    Calculates the pairwise similarity of tasks based on their corresponding feature values.
    For matrix of shape (A, B, B), returns a matrix of shape (A, A) where each entry is the similarity
    between the tasks at the corresponding row indices.

    Parameters
    ----------
    feature_sim : np.array
        Matrix of feature similarities of shape (num_tasks, num_objects, num_objects).
    contexts_to_keep : list[np.array]
        List of arrays of contexts/tasks to keep.
    objects_to_keep : np.array
        Array of objects to keep for each task/context.
    num_objects_to_keep : np.array
        Array of number of objects to keep for each task/context.
    
    Returns
    -------
    np.array
        Matrix of pairwise task similarities.
    """

    task_ft_sim = []
    for i in range(len(contexts_to_keep)):
        for j in range(i+1,len(contexts_to_keep)):
            item_idxs = np.intersect1d(objects_to_keep[i][:num_objects_to_keep[i]],objects_to_keep[j][:num_objects_to_keep[j]])
            item_idxs = np.meshgrid(item_idxs,item_idxs)
            features_i = feature_sim[contexts_to_keep[i],item_idxs[0],item_idxs[1]][np.triu_indices(item_idxs[0].shape[0],1)]
            features_j = feature_sim[contexts_to_keep[j],item_idxs[0],item_idxs[1]][np.triu_indices(item_idxs[0].shape[0],1)]
            task_ft_sim.append(fast_correlation_vec_vec(features_i,features_j))
    return np.array(task_ft_sim)


@njit(fastmath=False)
def fast_correlation_vec_vec(vec1: np.array, vec2: np.array, replace_nan=np.nan) -> float:
    """
    Calculates the correlation between two vectors.

    Parameters
    ----------
    vec1 : np.array
        First vector.
    vec2 : np.array
        Second vector.
    replace_nan : float, optional
        Value to replace NaN values with, by default np.nan.
    
    Returns
    -------
    float
        Correlation between the two vectors.
    """
    x_mean = np.nanmean(vec1)
    y_mean = np.nanmean(vec2)
    xy_corr_numerator = 0
    corrs_denomenator_x, corrs_denomenator_y = 0, 0
    for i in range(len(vec1)):
        if np.isnan(vec1[i]) or np.isnan(vec2[i]):
            if np.isnan(replace_nan):
                continue
            else:
                xval=0-x_mean
                yval=0-y_mean
        else:
            xval = vec1[i]-x_mean
            yval = vec2[i]-y_mean
        xy_corr_numerator += xval*yval
        corrs_denomenator_x += xval*xval
        corrs_denomenator_y += yval*yval
    if corrs_denomenator_x*corrs_denomenator_y==0:
        return replace_nan
    return xy_corr_numerator/np.sqrt(corrs_denomenator_x*corrs_denomenator_y)


@njit(parallel=True,fastmath=True)
def fast_correlation_vec_vec_bootstrap(vec1: np.array, vec2: np.array,
                                       n_sims: int, replace_nan = np.nan) -> tuple[float, np.array]:
    """
    Calculates the correlation between two vectors,
    using a bootstrap procedure to estimate the null distribution.

    Parameters
    ----------
    vec1 : np.array
        First vector.
    vec2 : np.array
        Second vector.
    replace_nan : float, optional
        Value to replace NaN values with, by default np.nan.
    
    Returns
    -------
    float
        Correlation between the two vectors.
    """
    correlation = fast_correlation_vec_vec(vec1,vec2,replace_nan=replace_nan)
    null_correlations = np.zeros((n_sims,))
    for i in prange(n_sims):
        idxs = np.random.permutation(len(vec2))
        null_correlations[i] = fast_correlation_vec_vec(vec1,vec2[idxs])
    return correlation, null_correlations


""" Helper functions for calculating similarity matrices"""

@njit(parallel=True,fastmath=True)
def fast_cosine_matrix_2d(mat: np.array) -> np.array:
    """
    Calculates the cosine similarity between all pairs of vectors in a matrix.
    For a matrix of shape (A, B), returns a matrix of shape (A, A) where each entry is the cosine similarity
    of the vectors at the corresponding row indices.

    Parameters
    ----------
    mat : np.array
        Matrix of vectors to calculate cosine similarity between.
    
    Returns
    -------
    np.array
        Matrix of pairwise cosine similarities.
    """
    new_mat = np.zeros((mat.shape[0],mat.shape[0]))
    for i in range(len(mat)):
        for j in range(i+1,len(mat)):
            new_mat[i,j] = np.vdot(mat[i],mat[j])/(norm(mat[i])*norm(mat[j]))-1
            new_mat[j,i] = new_mat[i,j]
    return new_mat


@njit(parallel=True,fastmath=True)
def fast_cosine_matrix_3d(mat: np.array) -> np.array:
    """
    Calculates the cosine similarity between all pairs of vectors for each context in a matrix.
    For a matrix of shape (A, B, C), returns a matrix of shape (A, B, B), where each entry is the 2D cosine
    similarity matrix for each of the A contexts.

    Parameters
    ----------
    mat : np.array
        Matrix of vectors to calculate cosine similarity between.
    
    Returns
    -------
    np.array
        Matrix of pairwise cosine similarities for each context.
    """
    new_mat = np.zeros((mat.shape[0],mat.shape[1],mat.shape[1]))
    for c in range(len(mat)):
        for i in range(mat.shape[1]):
            for j in range(i+1,mat.shape[1]):
                new_mat[c,i,j] = np.vdot(mat[c,i],mat[c,j])/(norm(mat[c,i])*norm(mat[c,j]))-1
                new_mat[c,j,i] = new_mat[c,i,j]
    return new_mat










@njit(fastmath=True)
def fast_correlation_vec_mat(vec,mat,replace_nan=np.nan):
    corrs = np.zeros((mat.shape[1],))
    for i in prange(mat.shape[1]):
        corrs[i] = fast_correlation_vec_vec(vec,mat[:,i],replace_nan)
    return corrs
