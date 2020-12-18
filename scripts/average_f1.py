import sys
import pandas as pd
import os
import argparse

seeds = [1, 2, 3, 4, 5]
#langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'tr', 'ur', 'vi', 'zh']

def tag_results(filename):
  seeded_results = []
  for seed in seeds:
    results = {}
    c_f = filename + "_s{}/test_results.txt".format(seed)
    if os.path.exists(c_f):
        with open(c_f, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("language"):
                    lan = line.split("=")[1]
                elif line.startswith("f1"):
                    f1 = line.split(" = ")[1]
                    results[lan] = float(f1)*100
    
    #print(results)
    #print(sum(results.values()) / len(results))
    seeded_results.append(results)
  
  df = pd.DataFrame(seeded_results)
  print(df)
  print(df.mean(axis=1))
  print("ave over random seeds:", df.mean(axis=1).mean())
  
  #print(df.mean(axis=1))
  #print(pd.DataFrame(df, columns=langs).mean(axis=0).mean())
  df.to_csv(filename+".csv")

def classify_results(filename):
  seeded_results = []
  for seed in seeds:
    results = {}
    c_f = filename + "_s{}/test_results.txt".format(seed)
    if os.path.exists(c_f):
        with open(c_f, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0: continue
                if line.startswith("total"):
                    continue
                else:
                    toks = line.split("=")
                    lan = toks[0]
                    acc = toks[1]
                    results[lan] = float(acc)*100
    
    #print(results)
    #print(sum(results.values()) / len(results))
    seeded_results.append(results)
  
  df = pd.DataFrame(seeded_results)
  print(df)
  print(df.mean(axis=1))
  print("ave over random seeds:", df.mean(axis=1).mean())
  
  #print(df.mean(axis=1))
  #print(pd.DataFrame(df, columns=langs).mean(axis=0).mean())
  df.to_csv(filename+".csv")

def qa_results(filename, do_final=1):
  seeded_results = []
  seeded_exact_results = []
  for seed in seeds:
    results = {}
    exact_results = {}
    c_f = filename + "_s{}/predictions/xquad/xquad_test_results_{}.txt".format(seed, do_final)
    if os.path.exists(c_f):
        with open(c_f, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("language"):
                    lan = line.split(" = ")[1]
                elif line.startswith("f1"):
                    f1 = line.split(" = ")[1]
                    results[lan] = float(f1)
                elif line.startswith("exact"):
                    f1 = line.split(" = ")[1]
                    exact_results[lan] = float(f1)
    
    #print(results)
    #print(sum(results.values()) / len(results))
    seeded_results.append(results)
    seeded_exact_results.append(exact_results)
  
  df = pd.DataFrame(seeded_results)
  exact_df = pd.DataFrame(seeded_exact_results)
  combined_df = pd.concat([df, exact_df], keys=('f1','exact f1'))
  print(combined_df)
  print(combined_df.mean(axis=1))
  print("xquad ave f1 over random seeds:", combined_df.xs('f1').mean(axis=1).mean())
  print("xquad ave exact f1 over random seeds:", combined_df.xs('exact f1').mean(axis=1).mean())
  
  #print(df.mean(axis=1))
  #print(pd.DataFrame(df, columns=langs).mean(axis=0).mean())
  combined_df.to_csv(filename+"_xquad_{}.csv".format(do_final))

  seeded_results = []
  seeded_exact_results = []
  for seed in seeds:
    results = {}
    exact_results = {}
    c_f = filename + "_s{}/predictions/mlqa/mlqa_test_results_{}.txt".format(seed, do_final)
    if os.path.exists(c_f):
        with open(c_f, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("language"):
                    lan = line.split(" = ")[1]
                elif line.startswith("f1"):
                    f1 = line.split(" = ")[1]
                    results[lan] = float(f1)
                elif line.startswith("exact"):
                    f1 = line.split(" = ")[1]
                    exact_results[lan] = float(f1)
    
    #print(results)
    #print(sum(results.values()) / len(results))
    seeded_results.append(results)
    seeded_exact_results.append(exact_results)
  
  df = pd.DataFrame(seeded_results)
  exact_df = pd.DataFrame(seeded_exact_results)
  combined_df = pd.concat([df, exact_df], keys=('f1','exact f1'))
  print(combined_df)
  print(combined_df.mean(axis=1))
  print("mlqa ave f1 over random seeds:", combined_df.xs('f1').mean(axis=1).mean())
  print("mlqa ave exact f1 over random seeds:", combined_df.xs('exact f1').mean(axis=1).mean())
  
  #print(df.mean(axis=1))
  #print(pd.DataFrame(df, columns=langs).mean(axis=0).mean())
  combined_df.to_csv(filename+"_mlqa_{}.csv".format(do_final))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="results aggregate")
  parser.add_argument("--task", type=str, default="tag", help="[tag|classify|qa]")
  parser.add_argument("--f", type=str, default=None, help="base filename")
  args = parser.parse_args()

  if args.task == "tag":
    tag_results(args.f)
  elif args.task == "classify":
    classify_results(args.f)
  elif args.task == "qa":
    qa_results(args.f)

