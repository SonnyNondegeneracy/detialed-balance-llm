####################################
# idea-agent/funsearch-version/search/run.py                       
# 2025/11/26更新                    
# 用IdeaSearcher类来运行寻找势函数
# 在线离线方法交替运行
####################################
from IdeaSearch import IdeaSearcher
from evaluator import evaluate
import time
import os
from config import api_path,output_path,total_cycles,island_num,N_online,N_offline,online_cycle_epochs,offline_cycle_epochs,online_models,offline_models,online_translator,offline_translator,initial_ideas,examples_num,sample_temperature,model_sample_temperature,model_assess_window_size,hand_over_threshold,model_assess_average_order,system, prologue, epilogue, filter

assert len(online_models) == len(offline_models)

os.makedirs(output_path,exist_ok=True)
file_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
data_path = f'{output_path}/{file_name}'

####################################
def online_evaluate(idea: str) -> tuple:
    return evaluate(idea, translator=online_translator)
def offline_evaluate(idea: str) -> tuple:
    return evaluate(idea, translator=offline_translator)
####################################

def main():
    
    ideasearcher = IdeaSearcher()
    
    # load models
    ideasearcher.set_api_keys_path(api_path)
    # set minimum required parameters
    ideasearcher.set_program_name("Potential Function Search")
    ideasearcher.set_database_path(data_path)

    # ideasearcher.set_crossover_func(None)
    # ideasearcher.set_mutation_func(None)
    ideasearcher.set_system_prompt(system)
    ideasearcher.set_prologue_section(prologue)
    ideasearcher.set_epilogue_section(epilogue)
    ideasearcher.set_filter_func(filter)
    ideasearcher.set_examples_num(examples_num)
    ideasearcher.set_sample_temperature(sample_temperature)
    ideasearcher.add_initial_ideas(initial_ideas)
    ideasearcher.set_model_sample_temperature(model_sample_temperature)
    ideasearcher.set_model_assess_window_size(model_assess_window_size)
    ideasearcher.set_hand_over_threshold(hand_over_threshold)
    ideasearcher.set_model_assess_average_order(model_assess_average_order)
    ideasearcher.set_record_prompt_in_diary(True)
    ideasearcher.set_models(online_models)
    ideasearcher.set_model_assess_initial_score(20.0)
    ideasearcher.set_evaluate_func(online_evaluate)

    # add a dozen islands(Song et. al. 2510.08317, expert mode)
    for _ in range(island_num):
        ideasearcher.add_island()
    
    # Evolve for 15 cycles, 10 epochs on each island per cycle with ideas repopulated at the end
    for cycle in range(total_cycles):
        if cycle % (N_online + N_offline) < N_online:
            ideasearcher.set_models(online_models)
            ideasearcher.set_model_assess_initial_score(20.0)
            ideasearcher.set_evaluate_func(online_evaluate)
            ideasearcher.run(online_cycle_epochs)
        else:
            ideasearcher.set_models(offline_models)
            ideasearcher.set_model_assess_initial_score(20.0)
            ideasearcher.set_evaluate_func(offline_evaluate)
            ideasearcher.run(offline_cycle_epochs)
        ideasearcher.repopulate_islands()
        
        best_idea = ideasearcher.get_best_idea()
        best_score = ideasearcher.get_best_score()
        print(
            f"【第{cycle+1}轮】"
            f"目前最高得分{best_score:.2f}，这个idea是：\n"
            "==========================================================\n"
            f"{filter(best_idea)}\n"
            "==========================================================\n"
        )


if __name__ == "__main__":
    main()