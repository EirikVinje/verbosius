import os
import time
import config as config



def main(starttime, endtime, n_chunks, chunksize, dataset, user, chunkdist_n):

    path = f"/home/{user}/project/<gitlab_repo>/"
    
    markdown_file = f"run_{dataset}_{chunkdist_n}_{starttime}.md"

    path_to_markdown = os.path.join(path, markdown_file)

    with open(path_to_markdown, "w") as f:

        f.write(f"# Run on chunkdist : {chunkdist_n} \n")
        f.write(f"## Dataset: {dataset} \n")
        f.write(f"## Starttime: {starttime} \n")
        f.write(f"## Endtime: {endtime} \n")
        f.write(f"## Number of chunks: {n_chunks} \n")
        f.write(f"## Chunksize: {chunksize} \n")

        f.write(f"Parameters: \n")
        f.write(f"\n")

        f.write(" MAX_FEATURES ", config.MAX_FEATURES)
        f.write(f"\n")
        f.write(" MAX_DF ", config.MAX_DF)
        f.write(f"\n")
        f.write(" MIN_DF ", config.MIN_DF)
        f.write(f"\n")
        f.write(" NUMBER_OF_CLAUSES ", config.NUMBER_OF_CLAUSES) 
        f.write(f"\n")
        f.write(" S ", config.S) 
        f.write(f"\n")
        f.write(" T ", config.T)
        f.write(f"\n")
        f.write(" TM_EPOCHS ", config.TM_EPOCHS) 
        f.write(f"\n")
        f.write(" ERROR_MAX_FEATURES ", config.ERROR_MAX_FEATURES) 
        f.write(f"\n")
        f.write(" ERROR_NUMBER_OF_CLAUSES ", config.ERROR_NUMBER_OF_CLAUSES) 
        f.write(f"\n")
        f.write(" ERROR_S ", config.ERROR_S)
        f.write(f"\n")
        f.write(" ERROR_T ", config.ERROR_T)
        f.write(f"\n")
        f.write(" ERROR_MAX_DF ", config.ERROR_MAX_DF) 
        f.write(f"\n")
        f.write(" ERROR_MIN_DF ", config.ERROR_MIN_DF)
        f.write(f"\n")
        f.write(" SKB_score_func ", config.SKB_score_func) 
        f.write(f"\n")
        f.write(" STOPWORDS", config.STOPWORDS)
        f.write(f"\n")
        f.write(" N_JOBS ", config.N_JOBS)
        f.write(f"\n")
        f.write(" EARLY_STOP_ACC", config.EARLY_STOP_ACC)
        f.write(f"\n")
        f.write(" error_chunk", config.error_chunk)
        f.write(f"\n")
        f.write(" n_badtexts", config.n_badtexts)
        f.write(f"\n")
        f.write(" CV_MAX_FEATURES", config.CV_MAX_FEATURES)
        f.write(f"\n")
        f.write(" N_GRAM_RANGE", config.N_GRAM_RANGE)
        f.write(f"\n")
        f.write(" LITERAL_BUDGET", config.LITERAL_BUDGET)
        f.write(f"\n")
        f.write(" ERROR_LITERAL_BUDGET  ", config.ERROR_LITERAL_BUDGET)  




                



    os.system()


if __name__ == "__main__":

    main()