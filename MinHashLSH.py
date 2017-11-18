import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib


def create_signature_matrix(data, permutation):
    rows, cols, sigrows = len(data), len(data[0]), permutation
    #Intializing the signature matrix
    sig_mat_shape = (sigrows, cols)
    sig_mat = np.zeros(sig_mat_shape, dtype=np.int)
    sig_val = 0
    permutation_list = list(range(0, len(data)))
    for  i in range(permutation):
        perm_index = np.random.permutation(permutation_list)
        for c in range(cols):
            for r in range(len(data)):
                if data[perm_index[r]][c] == 1:
                    sig_val = r
                    break
            sig_mat[i][c] = sig_val
    return sig_mat

def get_b_r_tuples(permutation):
    b_r_tuples = []
    for b in range(permutation):
        for r in range(permutation):
            if b * r == permutation:
                b_r_tuples.append((b,r))
    return b_r_tuples

def get_b_values(permutation):
    b_list = []
    for b in range(permutation):
        for r in range(permutation):
            if b * r == permutation:
                b_list.append(b)
    return b_list

def get_best_b_r_values(threshold, permutation):
    similar_set = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    best_probabilities = []
    max_slope = 0
    best_b_r = ()
    for b,r in get_b_r_tuples(permutation):
        probabilities = []
        for similarity in similar_set:
            probability = 1 - math.pow((1 - math.pow(similarity,r)),b)
            probabilities.append(probability)
        slope_range = probabilities[4]-probabilities[2]
        if slope_range > max_slope:
            max_slope = slope_range
            best_probabilities = probabilities
            best_b_r =(b,r)
    graph_text = "Permutation = "+ str(permutation)
    plt.plot(similar_set,best_probabilities)
    plt.title(graph_text)
    plt.xlabel("Jaccard Similarities")
    plt.ylabel("Candidate Probabilities")
    plt.show()
    return best_b_r

def find_candiate_pairs(bands,rows,signature_matrix):
    start = 0
    stop = rows
    cols = len(signature_matrix[0])
    candidate_pairs = set()
    for b in range(bands):
        for i in range(cols):
            for j in range(i+1,cols):
                if (i,j) in candidate_pairs:
                    continue
                column_one = signature_matrix[start:stop,i].flatten()
                column_two = signature_matrix[start:stop,j].flatten()
                column_one_hash = hash(tuple(column_one)) % 10000
                column_two_hash = hash(tuple(column_two)) % 10000
                if column_one_hash == column_two_hash:
                    candidate_pairs.add((i,j))
        start += rows
        stop += rows
    print("Total Candidate Pairs " + str(len(candidate_pairs)))
    return candidate_pairs


def find_jaccard_total_similarity(data):
    no_objects = len(data[0])
    no_items = len(data)
    sims_dictionary = dict()
    for i in range(no_objects):
        for j in range(i + 1, no_objects):
            tuple_key = (i, j)
            set1 = data[:, [i]].flatten().astype(np.bool)
            set2 = data[:, [j]].flatten().astype(np.bool)
            numerator = np.logical_and(set1,set2)
            denominator = np.logical_or(set1,set2)
            sim = numerator.sum() / denominator.sum()
            sims_dictionary[tuple_key] = sim
    return sims_dictionary


def find_jaccard_signature_similarity(signature_matrix):
    similarity_dictionary = dict()
    cols = len(signature_matrix[0])
    for i in range(cols):
        for j in range(i+1, cols):
            numerator_counter = 0
            denominator_count = len(signature_matrix)
            for r in range(len(signature_matrix)):
                if(signature_matrix[r,i] == signature_matrix[r,j]):
                    numerator_counter += 1
            jaccard_probability = numerator_counter / denominator_count
            similarity_dictionary[(i,j)] = jaccard_probability
    return similarity_dictionary

def find_fp_fn(signature_matrix,candidate_pairs, data):
     jacc_sig_similarity = find_jaccard_signature_similarity(signature_matrix)
     sig_false_positives = 0
     sig_false_negatives = 0
     for i in list(candidate_pairs):
        if jacc_sig_similarity[i] < 0.3:
            sig_false_positives += 1
            candidate_pairs.remove(i)
     for key in jacc_sig_similarity.keys():
         if jacc_sig_similarity[key] > 0.3 and key not in candidate_pairs:
             sig_false_negatives +=1
     similar_signatures = len(candidate_pairs)
     jacc_similarity = find_jaccard_total_similarity(data)
     false_positives = 0
     false_negatives = 0
     for i in list(candidate_pairs):
         if jacc_similarity[i] < 0.3:
             false_positives += 1
             candidate_pairs.remove(i)
     for key in jacc_similarity.keys():
         if jacc_similarity[key] > 0.3 and key not in candidate_pairs:
                false_negatives +=1
     similar_pairs = len(candidate_pairs)
     return false_positives, false_negatives, sig_false_positives, sig_false_negatives, similar_signatures, similar_pairs

def perform_collaborative_filtering(data, permutation, threshold):
    signature_matrix = create_signature_matrix(data, permutation)
    b, r = get_best_b_r_values(threshold, permutation)
    candidate_pairs = find_candiate_pairs(b, r, signature_matrix)
    fp, fn, sig_fp, sig_fn, sim_sig, sim_pair = find_fp_fn(signature_matrix, candidate_pairs, data)
    print("Permutation Level - " + str(permutation))
    print("Best b value" + str(b))
    print("Best r value" + str(r))
    print("FP - Signature " + str(sig_fp), "FN - Signature " + str(sig_fn))
    print("Similar pair of Signatures - " + str(sim_sig))
    print("FP - Total " + str(fp), "FN - Total " + str(fn))
    print("Similar Pairs - " + str(sim_pair))

def perform_filtering_complete(data,permutation, threshold):
    signature_matrix = create_signature_matrix(data, permutation)
    b_r_tuples = get_b_r_tuples(permutation)
    fp_list =[]
    fn_list = []
    sig_fp_list = []
    sig_fn_list = []
    b_list = []
    for b, r in b_r_tuples:
        candidate_pairs = find_candiate_pairs(b,r,signature_matrix)
        fp, fn, sig_fp, sig_fn, sim_sig, sim_pairs= find_fp_fn(signature_matrix,candidate_pairs,data)
        fp_list.append(fp)
        fn_list.append(fn)
        sig_fp_list.append(sig_fp)
        sig_fn_list.append(sig_fn)
        b_list.append(b)
    generate_plots(sig_fp_list,sig_fn_list,b_list,"No of Bands Vs Signature FP and FN")
    generate_plots(fp_list,fn_list,b_list,"No of Bands Vs FP and FN")
    print(b_list,"B List")
    print(fp_list,"FP List")
    print(fn_list, "FN List")

def generate_plots(fp_list, fn_list, b_list, graph_text):
    plt.plot(b_list, fp_list, color='orange', label="FP")
    plt.plot(b_list, fn_list, color="blue", label="FN")
    plt.legend()
    plt.title(graph_text)
    plt.xlabel("No of Bands")
    plt.ylabel("FP, FN")
    plt.savefig(graph_text+".png")
    plt.show()


if __name__ == '__main__':
    raw = []
    with open('data.txt', 'r') as f:
       for line in f:
            raw.append(map(int, line.split(',')))
    data_f = pd.DataFrame(raw)
    data = np.array(data_f)
    # Perform collaborative filtering for Permutations = 100
    perform_collaborative_filtering(data,100,0.30)

    # Perform collaborative filtering for Permutations = 500
    #perform_collaborative_filtering(data,500,0.30)

    # To calculate the candidate pairs and FPs and FNs for all values of Bands and generate plots
    #perform_filtering_complete(data,500,0.30)
