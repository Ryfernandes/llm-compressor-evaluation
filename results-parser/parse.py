class Parser():
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def parse(self, dirs: list[str]):
        """
        This method will parse the results from lm-eval and lighteval tasks and save the results to a single, structured json file.
        The method will receive dirs as an argument, which is a list of paths (relative or absolute) to directories containing the
        results of evaluation tasks. There may be files in this directory that are not results (such as logging or samples data), but
        all JSON files in the directory will be results files, all from a single lm-eval or lighteval task. The method should, for each
        directory path, recursively parse all JSON files in the directory and its subdirectories, scraping the following information:

        - The name of the task (e.g. "gsm8k_platinum_cot_llama" or "ifeval")
        - The aggregate metrics of interest (e.g. "exact_match,strict-match" or "exact_match,flexible_extract")
        - The model name
        - The inference parameters that were set (e.g. "max_length", "do_sample", "temperature", "top_p", "top_k", "max_gen_toks", "seed")
        - Version information (lm_eval, transformers, the task that ran)
        - The date and time of the evaluation run
        - How long the evaluation run took (in seconds)
        - The number of samples that were evaluated
        - The filename of the JSON file that was parsed
        
        These results should be aggregated for all results files into a JSON file and written to output_dir, with the filename "parsed_results_[timestamp].json"
        """

        raise NotImplementedError()