import os
import random
import io
import numpy as np
import scipy as sp
import pandas as pd
from collections import Counter
from os import listdir
from os.path import isfile, join, isdir
from random import shuffle
import fitz
import re

def load_data_recursive(root_dir, max_n_instances_per_class=None, max_n_lines=None, min_instance_len_in_characters=50):
    min_instance_len = 10
    dirnames = [f for f in listdir(root_dir) if isdir(join(root_dir, f))]
    documents = []
    targets = []
    for dirname in dirnames:
        dirname_path = join(root_dir, dirname)
        filenames = [filename for filename in listdir(dirname_path) if isfile(join(dirname_path, filename))]
        for count_instances, filename in enumerate(filenames):
            if max_n_instances_per_class is not None and count_instances >= max_n_instances_per_class:
                break
            filename_path = join(root_dir, dirname, filename)
            with open(filename_path, 'r',encoding='utf8') as file_handle:
                document = []
                for count_n_lines, line in enumerate(file_handle):
                    if max_n_lines is not None and count_n_lines >= max_n_lines:
                        break
                    document.append(line.strip())
                document = ' '.join(document) 
                document = ''.join([i for i in document if not i.isdigit()])
                document = ''.join([i for i in document if ord(i) < 128])
                if len(document) > min_instance_len_in_characters:
                    documents.append(document)
                    targets.append(dirname)
    return documents, targets

def load_data_recursive_pdf(root_dir, max_n_instances_per_class=None, min_instance_len_in_characters=1):
    dirnames = [f for f in listdir(root_dir) if isdir(join(root_dir, f))]
    documents = []
    pgct = []
    targets = []
    imgct = []
    for dirname in dirnames:
        dirname_path = join(root_dir, dirname)
        filenames = [filename for filename in listdir(dirname_path) if isfile(join(dirname_path, filename))]
        for count_instances, filename in enumerate(filenames):
            if max_n_instances_per_class is not None and count_instances >= max_n_instances_per_class:
                break
            if filename.endswith('.pdf'):
#                print(filename)
                filename_path = join(root_dir, dirname, filename)
                doc=fitz.open(filename_path)
#                 pgct.append(doc.page_count)
                token_text=''
                for page in doc:
                    text = page.get_text('text').strip()
                    token_text = token_text+text
                    text=''
                if len(token_text)>min_instance_len_in_characters:
                    documents.append(token_text)
                    targets.append(dirname)
    documents=[re.sub('[\n\.]+', '',r).strip() for r in documents]
    return documents, targets

def create_multiclass_documents(rootdir,n_documents=1,n_classes_min=3,n_classes_max=7,random_seed=0):
    random.seed=random_seed
    all_classes = os.listdir(rootdir)
    documents=[]
    targets=[]
    for n in range(n_documents):
        lines=[]
        doc_targets=[]
        start=end=ct=0
        random_classes=random.sample(all_classes,random.randrange(n_classes_min,n_classes_max,1))
        for a in all_classes:
            if a in random_classes:
                start=ct
                doc=fitz.open(rootdir+'/'+a+'/'+random.choice(os.listdir(rootdir +'/'+a)))

                token_text=''
                for page in doc:
                    text = page.get_text('text').strip()
                    token_text = token_text+text
                    text=''

                if len(token_text) > 0:
                    for line in io.StringIO(token_text):           
                        clean_line=re.sub('[\d]+', '',line.strip('\n'))
                        clean_line=re.sub('^\s+', '',clean_line.strip())
                        if len(clean_line)>1:
                            lines.append(clean_line)
                            ct+=1
                    end=ct
                    doc_targets.append((a,start,end-1))
                else:
                    doc_targets.append((a,-1,-1))
            else:
                doc_targets.append((a,-1,-1))
        documents.append(lines)
        targets.append(doc_targets)   
    return documents, targets


def document_line_lengths(root_dir):
    d1 = {}
    dirnames = [f for f in listdir(root_dir) if isdir(join(root_dir, f))]
    for dirname in dirnames:
        class_lengths=[]
        dirname_path = join(root_dir, dirname)
        filenames = [filename for filename in listdir(dirname_path) if isfile(join(dirname_path, filename))]
        for count_instances, filename in enumerate(filenames):
            lines=[]
            if filename.endswith('.pdf'):
                filename_path = join(root_dir, dirname, filename)
                doc=fitz.open(filename_path) 
                token_text=''
                for page in doc:
                    text = page.get_text('text').strip()
                    token_text = token_text+text
                    text=''
                if len(token_text) > 0:
                    for ct,line in enumerate(io.StringIO(token_text)):           
                        clean_line=re.sub('[\d]+', '',line.strip('\n'))
                        clean_line=re.sub('^\s+', '',clean_line.strip())
                        if len(clean_line)>1:
                            lines.append(clean_line)
        class_lengths.append(len(lines))
    d1[dirname] = class_lengths
    return d1


def load_data(root_dir, filename):
    documents = []
    targets = []
    filename_path = join(root_dir, filename)
    with open(filename_path, 'r') as file_handle:
        for line in file_handle:
            document = line.strip().lower() 
            values = document.split()
            target = int(values[0])
            document = ' '.join(values[1:])
            documents.append(document)
            targets.append(target)
    return documents, targets

def load_reuters_data(root_dir, min_num_documents=1, max_n_instances_per_class=500):
    filename_path = join(root_dir, 'multi-labels.tsv')
    df = pd.read_csv(filename_path,sep='\t')
    multilabel_targets =[[c for c in a.split("'") if '[' not in c and ']' not in c and ',' not in c] for a in df['label']] 
    targets = ['_'.join(sorted(target[0:2])) for target in multilabel_targets]
    documents = [''.join([c for c in document.replace('\n',' ').replace("\'s","'s").replace("\'t","'t") if not c.isdigit() and ord(c)<128 ]) for document in df['document']]
    target_counter = Counter(targets)
    valid_targets = [t for t in target_counter if target_counter[t] >= min_num_documents]
    valid_documents = [d for t,d in zip(targets, documents) if t in valid_targets]
    valid_multilabel_targets = [m for t,m in zip(targets, multilabel_targets) if t in valid_targets]
    valid_targets = [t for t,d in zip(targets, documents) if t in valid_targets]
    documents = valid_documents
    targets = valid_targets
    multilabel_targets = valid_multilabel_targets
    
    documents_list = [[d for d,t in zip(documents, targets) if t==current_target] for current_target in sorted(set(targets))] 
    multilabel_targets_list = [[m for m,t in zip(multilabel_targets, targets) if t==current_target] for current_target in sorted(set(targets))] 
    valid_documents = []
    valid_targets = []
    valid_multilabel_targets = []
    for document_list, multilabel_target_list, current_target in zip(documents_list, multilabel_targets_list, sorted(set(targets))):
        docs = document_list[:max_n_instances_per_class]
        valid_multilabel_targets.extend(multilabel_target_list[:max_n_instances_per_class])
        valid_documents.extend(docs)
        valid_targets.extend([current_target]*len(docs))
    documents,targets, multilabel_targets = valid_documents, valid_targets, valid_multilabel_targets

    return documents, targets, multilabel_targets
    
def load_wos_data(root_dir, size, level):
    documents = []
    if size == 'small':
        filename = 'WOS5736'
    elif size == 'medium':
        filename = 'WOS11967'
    elif size == 'large':
        filename = 'WOS46985'
    fname = 'X.txt'
    filename_path = join(root_dir, filename, fname)
    with open(filename_path, 'r') as file_handle:
        for line in file_handle:
            document = line.strip().lower() 
            documents.append(document)
    targets_list = []
    fname1 = 'YL1.txt'
    fname2 = 'YL2.txt'
    fname3 = 'Y.txt'
    fnames = [fname1, fname2, fname3]
    for fname in fnames:
        targets = []
        filename_path = join(root_dir, filename,fname)
        with open(filename_path, 'r') as file_handle:
            for line in file_handle:
                target = int(line.strip().lower() )
                targets.append(target)
        targets_list.append(targets)
    targets = []
    if level == 1:
        targets = targets_list[0]
    elif level == 2:
        for t1,t2 in zip(targets_list[0], targets_list[1]):
            if t1 ==1 and t2 > 8:
                targets.append('%d_%d'%(t1,t2-1))
            else:
                targets.append('%d_%d'%(t1,t2))
    elif level == 3:
        for t1,t2,t3 in zip(targets_list[0], targets_list[1], targets_list[2]):
            if t1 ==1 and t2 > 8:
                targets.append('%d_%d_%d'%(t1,t2-1,t3))
            else:
                targets.append('%d_%d_%d'%(t1,t2,t3))
    return documents, targets, targets_list