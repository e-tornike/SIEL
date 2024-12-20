#!/bin/bash
# This script extracts texts from downloaded PDF files using the running grobid server from 'setup_grobid.sh'

grobid_client processFulltextDocument --input ./0_pdfs/ --output ./1_xmls/ --n 1 --verbose --teiCoordinates --include_raw_citations