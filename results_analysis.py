#### Preanalysis the node degree results
import json
import torch
import math

def node_degree_results(edge_dict, results, test_hits, test_mrr, split_edge, node):
    if str(node) in edge_dict:
        if edge_dict[str(node)] not in results:
            results[edge_dict[str(node)]] = {}
            results[edge_dict[str(node)]]["hits"] = [test_hits]
            results[edge_dict[str(node)]]["mrr"] = [test_mrr]
            results[edge_dict[str(node)]]["number"] = 1
            results[edge_dict[str(node)]]["hit_num"] = [split_edge['test']['new'][node]["positive"].size(1)]
        else:
            results[edge_dict[str(node)]]["hits"].append(test_hits)
            results[edge_dict[str(node)]]["mrr"].append(test_mrr)
            results[edge_dict[str(node)]]["number"] += 1
            results[edge_dict[str(node)]]["hit_num"].append(split_edge['test']['new'][node]["positive"].size(1))
    else:
        if 0 not in results:
            results[0] = {}
            results[0]["hits"] = [test_hits]
            results[0]["mrr"] = [test_mrr]
            results[0]["number"] = 1
            results[0]["hit_num"] = [split_edge['test']['new'][node]["positive"].size(1)]
        else:
            results[0]["hits"].append(test_hits)
            results[0]["mrr"].append(test_mrr)
            results[0]["number"] += 1
            results[0]["hit_num"].append(split_edge['test']['new'][node]["positive"].size(1))

    return results

def result_coldwarm(results, split_num):
    log_results = {}
    hits_list = [f'hits@{K}' for K in [10,20,30,50]]
    for this_count in results:
        if int(this_count) == 0:
            if '0' not in log_results:
                log_results['0'] = {}
                for hits_name in hits_list:
                    log_results['0'][hits_name] = 0.0
                # log_results['0']["hits"] = 0.0
                log_results['0']["mrr"] = 0.0
                log_results['0']["auc"] = 0.0
                log_results['0']["edge_num"] = 0
                log_results['0']["node_num"] = results[this_count]["number"]                
            for i in range(len(results[this_count]["mrr"])):
                for hits_name in hits_list:
                    log_results['0'][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
                # log_results['0']["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
                log_results['0']["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
                log_results['0']["auc"] += results[this_count]["auc"][i]
                log_results['0']["edge_num"] += results[this_count]["hit_num"][i]
        elif int(this_count) > 0 and int(this_count) < (split_num+1):
            if f'1-{split_num}' not in log_results:
                log_results[f'1-{split_num}'] = {}
                # log_results[f'1-{split_num}']["hits"] = 0.0
                for hits_name in hits_list:
                    log_results[f'1-{split_num}'][hits_name] = 0.0
                log_results[f'1-{split_num}']["mrr"] = 0.0
                log_results[f'1-{split_num}']["auc"] = 0.0
                log_results[f'1-{split_num}']["edge_num"] = 0
                log_results[f'1-{split_num}']["node_num"] = results[this_count]["number"]
            else:
                log_results[f'1-{split_num}']["node_num"] += results[this_count]["number"]
            for i in range(len(results[this_count]["mrr"])):
                for hits_name in hits_list:
                    log_results[f'1-{split_num}'][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
                # log_results[f'1-{split_num}']["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
                log_results[f'1-{split_num}']["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
                log_results[f'1-{split_num}']["auc"] += results[this_count]["auc"][i]
                log_results[f'1-{split_num}']["edge_num"] += results[this_count]["hit_num"][i]
        else:
            if f'>{split_num}' not in log_results:
                log_results[f'>{split_num}'] = {}
                for hits_name in hits_list:
                    log_results[f'>{split_num}'][hits_name] = 0.0
                # log_results[f'>{split_num}']["hits"] = 0.0
                log_results[f'>{split_num}']["mrr"] = 0.0
                log_results[f'>{split_num}']["auc"] = 0.0
                log_results[f'>{split_num}']["edge_num"] = 0
                log_results[f'>{split_num}']["node_num"] = results[this_count]["number"]
            else:
                log_results[f'>{split_num}']["node_num"] += results[this_count]["number"]
            for i in range(len(results[this_count]["mrr"])):
                for hits_name in hits_list:
                    log_results[f'>{split_num}'][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
                # log_results[f'>{split_num}']["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
                log_results[f'>{split_num}']["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
                log_results[f'>{split_num}']["auc"] += results[this_count]["auc"][i]
                log_results[f'>{split_num}']["edge_num"] += results[this_count]["hit_num"][i]

        if "overall" not in log_results:
            log_results["overall"] = {}
            # log_results["overall"]["hits"] = 0.0
            for hits_name in hits_list:
                log_results["overall"][hits_name] = 0.0
            log_results["overall"]["mrr"] = 0.0
            log_results["overall"]["auc"] = 0.0
            log_results["overall"]["edge_num"] = 0
            log_results["overall"]["node_num"] = results[this_count]["number"]
        else:
            log_results["overall"]["node_num"] += results[this_count]["number"]
        for i in range(len(results[this_count]["mrr"])):
            for hits_name in hits_list:
                log_results["overall"][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
            # log_results["overall"]["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
            log_results["overall"]["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
            log_results["overall"]["auc"] += results[this_count]["auc"][i]
            log_results["overall"]["edge_num"] += results[this_count]["hit_num"][i]
        
    return log_results

def result_log(results):
    log_results = {}
    hits_list = [f'hits@{K}' for K in [10,20,30,50]]
    for this_count in results:
        if str(int(math.log2(int(this_count)+1))) not in log_results:
            log_results[str(int(math.log2(int(this_count)+1)))] = {}
            for hits_name in hits_list:
                log_results[str(int(math.log2(int(this_count)+1)))][hits_name] = 0.0
            # log_results[str(int(math.log2(int(this_count)+1)))]["hits"] = 0.0
            log_results[str(int(math.log2(int(this_count)+1)))]["mrr"] = 0.0
            log_results[str(int(math.log2(int(this_count)+1)))]["auc"] = 0.0
            log_results[str(int(math.log2(int(this_count)+1)))]["edge_num"] = 0
            log_results[str(int(math.log2(int(this_count)+1)))]["node_num"] = results[this_count]["number"]
        else:
            log_results[str(int(math.log2(int(this_count)+1)))]["node_num"] += results[this_count]["number"]
        for i in range(len(results[this_count]["mrr"])):
            for hits_name in hits_list:
                log_results[str(int(math.log2(int(this_count)+1)))][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
            # log_results[str(int(math.log2(int(this_count)+1)))]["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
            log_results[str(int(math.log2(int(this_count)+1)))]["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
            log_results[str(int(math.log2(int(this_count)+1)))]["auc"] += results[this_count]["auc"][i]
            log_results[str(int(math.log2(int(this_count)+1)))]["edge_num"] += results[this_count]["hit_num"][i]

        if "overall" not in log_results:
            log_results["overall"] = {}
            for hits_name in hits_list:
                log_results["overall"][hits_name] = 0.0
            # log_results["overall"]["hits"] = 0.0
            log_results["overall"]["mrr"] = 0.0
            log_results["overall"]["auc"] = 0.0
            log_results["overall"]["edge_num"] = 0
            log_results["overall"]["node_num"] = results[this_count]["number"]
        else:
            log_results["overall"]["node_num"] += results[this_count]["number"]
        for i in range(len(results[this_count]["mrr"])):
            for hits_name in hits_list:
                log_results["overall"][hits_name] += results[this_count][hits_name][i] * results[this_count]["hit_num"][i]
            # log_results["overall"]["hits"] += results[this_count]["hits"][i] * results[this_count]["hit_num"][i]
            log_results["overall"]["mrr"] += results[this_count]["mrr"][i] * results[this_count]["hit_num"][i]
            log_results["overall"]["auc"] += results[this_count]["auc"][i]
            log_results["overall"]["edge_num"] += results[this_count]["hit_num"][i]

    return log_results

def merge_results(log_results, all_results, overall_results):
    hits_list = [f'hits@{K}' for K in [10,20,30,50]]
    for key in log_results.keys():
        if key != "overall":
            if key not in all_results:
                all_results[key] = {}
                all_results[key]["node_num"] = int(log_results[key]["node_num"])
                all_results[key]["edge_num"] = int(log_results[key]["edge_num"])
                for hits_name in hits_list:
                    all_results[key][hits_name] = [log_results[key][hits_name]/ log_results[key]["edge_num"]]
                # all_results[key]['hits'] = [log_results[key]["hits"]/ log_results[key]["edge_num"]]
                all_results[key]['mrr'] = [log_results[key]["mrr"]/ log_results[key]["edge_num"]]
                all_results[key]['auc'] = [log_results[key]["auc"]/ log_results[key]["node_num"]]
            else:
                for hits_name in hits_list:
                    all_results[key][hits_name].append(log_results[key][hits_name]/ log_results[key]["edge_num"])
                all_results[key]['auc'].append(log_results[key]["auc"]/ log_results[key]["node_num"])
                all_results[key]['mrr'].append(log_results[key]["mrr"]/ log_results[key]["edge_num"])
        else:
            if "mrr" not in overall_results:
                overall_results['node_num'] = int(log_results[key]["node_num"])
                overall_results["edge_num"] = int(log_results[key]["edge_num"])
                for hits_name in hits_list:
                    overall_results[hits_name] = [log_results[key][hits_name]/ log_results[key]["edge_num"]]
                # all_results[key]['hits'] = [log_results[key]["hits"]/ log_results[key]["edge_num"]]
                overall_results['mrr'] = [log_results[key]["mrr"]/ log_results[key]["edge_num"]]
                overall_results['auc'] = [log_results[key]["auc"]/ log_results[key]["node_num"]]
            else:
                for hits_name in hits_list:
                    overall_results[hits_name].append(log_results[key][hits_name]/ log_results[key]["edge_num"])
                overall_results['auc'].append(log_results[key]["auc"]/ log_results[key]["node_num"])
                overall_results['mrr'].append(log_results[key]["mrr"]/ log_results[key]["edge_num"])
    
    return all_results, overall_results

def sp_results_coldwarm(these_results, split_num):
    all_results = {}
    overall_results = {}

    for results in these_results:
        log_results = result_coldwarm(results, split_num)
        all_results, overall_results = merge_results(log_results, all_results, overall_results)
        
    return all_results, overall_results

def sp_results_log(these_results):
    all_results = {}
    overall_results = {}

    for results in these_results:
        log_results = result_log(results)
        all_results, overall_results = merge_results(log_results, all_results, overall_results)
        
    return all_results, overall_results

def save_results(file, all_results, overall_results):
    for key in sorted(all_results.keys()):
        print_out_str = str(key) + ", " + str(all_results[key]["node_num"]) + ", " + str(all_results[key]["edge_num"])
        for this_K in [10,20,30,50]:
            this_res_hits = torch.tensor(all_results[key][f'hits@{this_K}'])
            print_out_str += f', hits@{this_K}: ' + f'{this_res_hits.mean():.2f} ± {this_res_hits.std():.2f}'
        this_res_mrr = torch.tensor(all_results[key]['mrr'])
        print_out_str += ", mrr: " + f'{this_res_mrr.mean():.2f} ± {this_res_mrr.std():.2f}'
        this_res_auc = torch.tensor(all_results[key]['auc'])
        print_out_str += ", auc: " + f'{this_res_auc.mean():.2f} ± {this_res_auc.std():.2f}'
        print(print_out_str+"\n")
        file.write(print_out_str+"\n")

    print_out_str = "Overall, " + str(overall_results["node_num"]) + ", " + str(overall_results["edge_num"])
    for this_K in [10,20,30,50]:
        this_res_hits = torch.tensor(overall_results[f'hits@{this_K}'])
        print_out_str += f', hits@{this_K}: ' + f'{this_res_hits.mean():.2f} ± {this_res_hits.std():.2f}'
    this_res_mrr = torch.tensor(overall_results['mrr'])
    print_out_str += ", mrr: " + f'{this_res_mrr.mean():.2f} ± {this_res_mrr.std():.2f}'
    this_res_auc = torch.tensor(overall_results['auc'])
    print_out_str += ", auc: " + f'{this_res_auc.mean():.2f} ± {this_res_auc.std():.2f}'
    print(print_out_str+"\n")
    file.write(print_out_str+"\n")


   