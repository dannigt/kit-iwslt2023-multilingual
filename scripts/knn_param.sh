# e.g. using all ACL dev bitext as datastore
# the number of target tokens in the datastore
# check preprocess.log for total target tokens in the datastore
# the minimum count is the target token count in preprocess.log + # total sentences (because of the target language token) + 1
declare -A tokmap=( ["de"]=14004 ["ar"]=13247 ["zh"]=10851 ["nl"]=13177 ["fr"]=15575 ["ja"]=12719 ["fa"]=12889 ["pt"]=13049 ["ru"]=13516 ["tr"]=12593 )
dstore_name=ACL2023dev