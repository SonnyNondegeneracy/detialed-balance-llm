import sys
############################################
api_path = 'INSERT_YOUR_API_PATH_HERE'
idesearch_result_path = './data/ideasearch_raw'
try:
    # Load discreption and Nikuradse data
    from load_data import load_data
except ImportError as e:
    print(f"Failed to import necessary modules. Please check the paths in sys.path. Error: {e}")
    sys.exit(1)
############################################
# Add necessary directories to sys.path to handle imports
import numpy as np
from os.path import sep as seperator
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter
import time
import os
import json

os.makedirs(idesearch_result_path,exist_ok=True)
models = ['gemini-pro', 'gpt-5-mini', 'gpt-5', 'qwen3', 'qwen-plus', 'gemini-2.5-flash', 'deepseek-v3', 'grok-4', 'doubao', 'gemini-2.5-pro']
translator = "gemini-2.5-flash"
ok_score = 100.0

def run_ideasearch(data_name, output_dir=idesearch_result_path):
    """Run IdeaSearch to fit the specified dataset and save results to the specified directory."""
    use_fuzzy = True
    output_dir = output_dir + f'/pmlb/' + data_name
    os.makedirs(output_dir,exist_ok=True)

    try:
        x, y, yerr, info = load_data(data_name,method = 'train', polish=False)
    except Exception as e:
        print(e)
        return (data_name, f"Failure", f'Error loading data: {e}')
    variable_names = info['feature_names']
    variable_units = info['feature_units']
    output_name = info['target_name']
    output_unit = info['target_unit']

    # build IdeaSearcher and IdeaSearchFitter instances
    ideasearcher = IdeaSearcher()
    if yerr is not None:
        data = {"x": x, "y": y, "error": yerr,}
    else:
        data = {"x": x, "y": y, }
    ideasearch_fitter = IdeaSearchFitter(data = data,
                        metric_mapping='logarithm',
                        perform_unit_validation = False, 
                        variable_names = variable_names, 
                        variable_units = variable_units, 
                        output_name = output_name,
                        output_unit = output_unit,
                        auto_polish=True,
                        auto_polisher='gemini-pro',
                        generate_fuzzy=True,
                        fuzzy_translator=translator,
                        constant_whitelist=['1','2','pi'],
                        constant_map={'1':1, '2':2, 'pi':np.pi},
                        result_path=output_dir
                        )

    file_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Set database path first, all file system operations of ideasearcher will be done in the database
    ideasearcher.set_database_path(f"{output_dir}{seperator}{file_name}")
    # bind ideasearch_fitter
    # prompt, evaluation function and initial_ideas will be outsourced to ideasearch_fitter
    ideasearcher.bind_helper(ideasearch_fitter)
    
    # set other necessary and optional parameters in ideasearcher
    ideasearcher.set_program_name(f"IdeaSearch-fit {data_name}")
    
    # --------------------------------------
    ideasearcher.set_samplers_num(3)
    ideasearcher.set_sample_temperature(1000.0)
    ideasearcher.set_model_sample_temperature(1000.0)
    ideasearcher.set_hand_over_threshold(-0.1)
    ideasearcher.set_evaluators_num(3)
    ideasearcher.set_examples_num(1)
    ideasearcher.set_generate_num(3)
    ideasearcher.set_record_prompt_in_diary(True)
    ideasearcher.set_shutdown_score(ok_score)
    # --------------------------------------

    ideasearcher.set_api_keys_path(api_path)
    ideasearcher.set_model_assess_average_order(15.0)
    ideasearcher.set_model_assess_initial_score(20.0)
    ideasearcher.set_models(models)
    
    # Start IdeaSearch
    island_num = 8
    cycle_num = 30
    unit_interaction_num = 10
    
    for _ in range(island_num):
        ideasearcher.add_island()
    
    flag = False
    for cycle in range(cycle_num):
        ideasearcher.set_filter_func(lambda idea: "")

        if cycle != 0:
            ideasearcher.repopulate_islands()
    
        ideasearcher.run(unit_interaction_num)
        
        # Use get_best_fit action to view the best fitting function in real-time
        best_fun = ideasearch_fitter.get_best_fit()
        print(best_fun)

        best_score = ideasearcher.get_best_score()
        if best_score >= ok_score:
            print(f'Reach the preset score {best_score}, end early')
            flag = True
            break
        
    if flag:
        return (data_name, f"Success", f'Best function: {best_fun}')
    else:
        return (data_name, f"Failure", f'Best function: {best_fun}')
    
if __name__ == "__main__":
    for _ in range(10):
        data_name = "nikuradse_2"
        run_ideasearch(data_name)