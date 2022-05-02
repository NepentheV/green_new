#! /bin/bash

cd ../raw_data_MT
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
gzip -d reviews_Movies_and_TV_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz	
gzip -d meta_Movies_and_TV.json.gz
