
import numpy as np
import json
import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from util.common import preprocess


#
# code heavily modified but originally from:
# https://www.kaggle.com/code/jampaniramprasad/arxiv-abstract-classification-using-roberta
#


sci_field_map = {'astro-ph': 'Astrophysics',
                'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
                'astro-ph.EP': 'Earth and Planetary Astrophysics',
                'astro-ph.GA': 'Astrophysics of Galaxies',
                'astro-ph.HE': 'High Energy Astrophysical Phenomena',
                'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
                'astro-ph.SR': 'Solar and Stellar Astrophysics',
                'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
                'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
                'cond-mat.mtrl-sci': 'Materials Science',
                'cond-mat.other': 'Other Condensed Matter',
                'cond-mat.quant-gas': 'Quantum Gases',
                'cond-mat.soft': 'Soft Condensed Matter',
                'cond-mat.stat-mech': 'Statistical Mechanics',
                'cond-mat.str-el': 'Strongly Correlated Electrons',
                'cond-mat.supr-con': 'Superconductivity',
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control',
                'econ.EM': 'Econometrics',
                'eess.AS': 'Audio and Speech Processing',
                'eess.IV': 'Image and Video Processing',
                'eess.SP': 'Signal Processing',
                'gr-qc': 'General Relativity and Quantum Cosmology',
                'hep-ex': 'High Energy Physics - Experiment',
                'hep-lat': 'High Energy Physics - Lattice',
                'hep-ph': 'High Energy Physics - Phenomenology',
                'hep-th': 'High Energy Physics - Theory',
                'math.AC': 'Commutative Algebra',
                'math.AG': 'Algebraic Geometry',
                'math.AP': 'Analysis of PDEs',
                'math.AT': 'Algebraic Topology',
                'math.CA': 'Classical Analysis and ODEs',
                'math.CO': 'Combinatorics',
                'math.CT': 'Category Theory',
                'math.CV': 'Complex Variables',
                'math.DG': 'Differential Geometry',
                'math.DS': 'Dynamical Systems',
                'math.FA': 'Functional Analysis',
                'math.GM': 'General Mathematics',
                'math.GN': 'General Topology',
                'math.GR': 'Group Theory',
                'math.GT': 'Geometric Topology',
                'math.HO': 'History and Overview',
                'math.IT': 'Information Theory',
                'math.KT': 'K-Theory and Homology',
                'math.LO': 'Logic',
                'math.MG': 'Metric Geometry',
                'math.MP': 'Mathematical Physics',
                'math.NA': 'Numerical Analysis',
                'math.NT': 'Number Theory',
                'math.OA': 'Operator Algebras',
                'math.OC': 'Optimization and Control',
                'math.PR': 'Probability',
                'math.QA': 'Quantum Algebra',
                'math.RA': 'Rings and Algebras',
                'math.RT': 'Representation Theory',
                'math.SG': 'Symplectic Geometry',
                'math.SP': 'Spectral Theory',
                'math.ST': 'Statistics Theory',
                'math-ph': 'Mathematical Physics',
                'nlin.AO': 'Adaptation and Self-Organizing Systems',
                'nlin.CD': 'Chaotic Dynamics',
                'nlin.CG': 'Cellular Automata and Lattice Gases',
                'nlin.PS': 'Pattern Formation and Solitons',
                'nlin.SI': 'Exactly Solvable and Integrable Systems',
                'nucl-ex': 'Nuclear Experiment',
                'nucl-th': 'Nuclear Theory',
                'physics.acc-ph': 'Accelerator Physics',
                'physics.ao-ph': 'Atmospheric and Oceanic Physics',
                'physics.app-ph': 'Applied Physics',
                'physics.atm-clus': 'Atomic and Molecular Clusters',
                'physics.atom-ph': 'Atomic Physics',
                'physics.bio-ph': 'Biological Physics',
                'physics.chem-ph': 'Chemical Physics',
                'physics.class-ph': 'Classical Physics',
                'physics.comp-ph': 'Computational Physics',
                'physics.data-an': 'Data Analysis, Statistics and Probability',
                'physics.ed-ph': 'Physics Education',
                'physics.flu-dyn': 'Fluid Dynamics',
                'physics.gen-ph': 'General Physics',
                'physics.geo-ph': 'Geophysics',
                'physics.hist-ph': 'History and Philosophy of Physics',
                'physics.ins-det': 'Instrumentation and Detectors',
                'physics.med-ph': 'Medical Physics',
                'physics.optics': 'Optics',
                'physics.plasm-ph': 'Plasma Physics',
                'physics.pop-ph': 'Popular Physics',
                'physics.soc-ph': 'Physics and Society',
                'physics.space-ph': 'Space Physics',
                'q-bio.BM': 'Biomolecules',
                'q-bio.CB': 'Cell Behavior',
                'q-bio.GN': 'Genomics',
                'q-bio.MN': 'Molecular Networks',
                'q-bio.NC': 'Neurons and Cognition',
                'q-bio.OT': 'Other Quantitative Biology',
                'q-bio.PE': 'Populations and Evolution',
                'q-bio.QM': 'Quantitative Methods',
                'q-bio.SC': 'Subcellular Processes',
                'q-bio.TO': 'Tissues and Organs',
                'q-fin.CP': 'Computational Finance',
                'q-fin.EC': 'Economics',
                'q-fin.GN': 'General Finance',
                'q-fin.MF': 'Mathematical Finance',
                'q-fin.PM': 'Portfolio Management',
                'q-fin.PR': 'Pricing of Securities',
                'q-fin.RM': 'Risk Management',
                'q-fin.ST': 'Statistical Finance',
                'q-fin.TR': 'Trading and Market Microstructure',
                'quant-ph': 'Quantum Physics',
                'stat.AP': 'Applications',
                'stat.CO': 'Computation',
                'stat.ME': 'Methodology',
                'stat.ML': 'Machine Learning',
                'stat.OT': 'Other Statistics',
                'stat.TH': 'Statistics Theory'}




def fetch_arxiv(data_path=None, test_size=.175, seed=1):

    print(f'fetching arxiv data... data_path: {data_path}, test_size: {test_size}, seed: {seed}')

    file_path = data_path + '/arxiv-metadata-oai-snapshot.json'
    print("file_path:", file_path)

    # Using `yield` to load the JSON file in a loop to prevent 
    # Python memory issues if JSON is loaded directly
    def get_raw_data():
        with open(file_path, 'r') as f:
            for thing in f:
                yield thing

    paper_titles = []
    paper_intro = []
    paper_type = []

    paper_categories = np.array(list(sci_field_map.keys())).flatten()

    metadata_of_paper = get_raw_data()
    for paper in metadata_of_paper:
        papers_dict = json.loads(paper)
        category = papers_dict.get('categories')
        try:
            try:
                year = int(papers_dict.get('journal-ref')[-4:])
            except:
                year = int(papers_dict.get('journal-ref')[-5:-1])

            if category in paper_categories and 2010<year<2021:
                paper_titles.append(papers_dict.get('title'))
                paper_intro.append(papers_dict.get('abstract'))
                paper_type.append(papers_dict.get('categories'))
        except:
            pass 

    papers_dataframe = pd.DataFrame({
        'title': paper_titles,
        'abstract': paper_intro,
        'categories': paper_type
    })

    # preprocess text
    #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.replace("\n"," "))
    #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.strip())

    papers_dataframe['text'] = papers_dataframe['title'] + '. ' + papers_dataframe['abstract']

    papers_dataframe['text'] = preprocess(
        papers_dataframe['text'],
        remove_punctuation=False,
        lowercase=True,
        remove_stopwords=False,
        remove_special_chars=True,
    )

    papers_dataframe['categories'] = papers_dataframe['categories'].apply(lambda x: tuple(x.split()))
    
    # Ensure the 'categories' column value counts are calculated and indexed properly
    categories_counts = papers_dataframe['categories'].value_counts().reset_index(name="count")
    
    # Filter for categories with a count greater than 250
    shortlisted_categories = categories_counts.query("count > 250")["categories"].tolist()
    #print("shortlisted_categories:", shortlisted_categories)

    # Choosing paper categories based on their frequency & eliminating categories with very few papers
    #shortlisted_categories = papers_dataframe['categories'].value_counts().reset_index(name="count").query("count > 250")["index"].tolist()
    papers_dataframe = papers_dataframe[papers_dataframe["categories"].isin(shortlisted_categories)].reset_index(drop=True)
    
    # Shuffle DataFrame
    papers_dataframe = papers_dataframe.sample(frac=1).reset_index(drop=True)

    # Sample roughtly equal number of texts from different paper categories (to reduce class imbalance issues)
    papers_dataframe = papers_dataframe.groupby('categories').head(250).reset_index(drop=True)

    # encode categories using MultiLabelBinarizer
    multi_label_encoder = MultiLabelBinarizer()
    multi_label_encoder.fit(papers_dataframe['categories'])
    papers_dataframe['categories_encoded'] = papers_dataframe['categories'].apply(lambda x: multi_label_encoder.transform([x])[0])

    papers_dataframe = papers_dataframe[["text", "categories", "categories_encoded"]]
    del paper_titles, paper_intro, paper_type
    print(papers_dataframe.head())

    # Convert encoded labels to a 2D array
    y = np.vstack(papers_dataframe['categories_encoded'].values)
    #y = papers_dataframe['categories_encoded'].values

    # Retrieve target names and number of classes
    target_names = multi_label_encoder.classes_
    num_classes = len(target_names)

    # split dataset into training and test set
    xtrain, xtest, ytrain, ytest = train_test_split(papers_dataframe['text'], y, test_size=test_size, random_state=seed)

    return xtrain.tolist(), ytrain, xtest.tolist(), ytest, target_names, num_classes
    


