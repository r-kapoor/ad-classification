import os
import sys
import utility.functions
import preprocess.embeddings
import sys
reload(sys)
sys.setdefaultencoding('utf8')

UNIGRAM_FILE = 'unigram-part-00000-v2.json'
TRAINING_FILE = 'training.csv'
CONFIG_FILE = 'ad_classification.ini'
RESOURCES_FOLDER = 'resources' #Should be present in the path of this file

#Preprocess the documents using the embeddings file

def create_doc_embeddings(data_file, embeddings_file, config):
    """
    Creates Doc Embeggings for the data file using the embeddings file
    """
    utility.functions.print_name()
    embeddings = preprocess.embeddings.create(embeddings_file)
    docs = preprocess.embeddings.get_doc_embeddings(data_file, embeddings, config)
    print docs

current_path = os.path.dirname(os.path.abspath(__file__))
resource = utility.functions.ResourcesFile(current_path, RESOURCES_FOLDER)
config = utility.functions.AdConfig(CONFIG_FILE)
create_doc_embeddings(resource.get(TRAINING_FILE), resource.get(UNIGRAM_FILE), config)