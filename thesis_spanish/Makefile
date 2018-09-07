# Makefile
# Author: Igor Ruiz-Agundez
# Affiliation: DeustoTech, Deusto Institute of Technology, University of Deusto
# Version: v.1.0


###
# TEX configuration
###

# Name of the TEX file to work with
FILE_TEX=dissertation_sp

###
# Backup configuration
###

# Date stamp style
DATESTAMP=`date +'%Y-%m-%d'`

###
# Authentication parameters
###

# Account tpye
ACCOUNTTYPE=GOOGLE

# Email
# EMAIL=your-email-with-google-account
EMAIL=your-email-with-google-account

# Password
# PASSWD=your-password
PASSWD=your-password

# Service type. Set it for documents
SERVICE=writely

# Source of the post request
SOURCE=deusto.es

###
# Google Docs resource id-s
###

# To get the resource id:
# Open the document with Google Docs
# Copy and past the document URL from your browser
# Example:
# https://docs.google.com/document/d/123XXX123XXX/edit?hl=en GB#
# In this example, the resource id will be:
# 123XXX123XXX

# tex file resource id
# TEX_GOOGLE_DOCS=123XXX123XXX
TEX_GOOGLE_DOCS=123XXX123XXX

# bib file resource id
# BIB_GOOGLE_DOCS=123XXX123XXX
BIB_GOOGLE_DOCS=123XXX123XXX


###
# make execution options
###

all: pdflatex

latex: clean
	latex ${FILE_TEX}.tex
	# Uncomment makeindex if the document contains an index
	makeindex ${FILE_TEX}.nlo -s nomencl.ist -o ${FILE_TEX}.nls
	bibtex ${FILE_TEX}
	latex ${FILE_TEX}.tex
	latex ${FILE_TEX}.tex
	dvipdfm ${FILE_TEX}.dvi
	# Backup tex, bib and generated pdf files
	# There is one backup per day
	mkdir -p time-machine/${DATESTAMP}
	cp ${FILE_TEX}.pdf time-machine/${DATESTAMP}/${FILE_TEX}.pdf

pdflatex: clean	
	pdflatex ${FILE_TEX}.tex
	# Uncomment makeindex if the document contains an index
	makeindex ${FILE_TEX}.nlo -s nomencl.ist -o ${FILE_TEX}.nls
	bibtex ${FILE_TEX}
	pdflatex ${FILE_TEX}.tex
	pdflatex ${FILE_TEX}.tex
	pdflatex ${FILE_TEX}.tex
	# Backup tex, bib and generated pdf files
	# There is one backup per day
	mkdir -p time-machine/${DATESTAMP}
	cp ${FILE_TEX}.pdf time-machine/${DATESTAMP}/${FILE_TEX}.pdf

rtf: clean	
	latex ${FILE_TEX}.tex
	# Uncomment makeindex if the document contains an index
	makeindex ${FILE_TEX}.nlo -s nomencl.ist -o ${FILE_TEX}.nls
	bibtex ${FILE_TEX}
	latex ${FILE_TEX}.tex
	latex ${FILE_TEX}.tex
	latex2rtf ${FILE_TEX}.tex
	# Backup tex, bib and generated rtf files
	# There is one backup per day
	mkdir -p time-machine/${DATESTAMP}
	cp ${FILE_TEX}.pdf time-machine/${DATESTAMP}/${FILE_TEX}.rtf

view:
	# Open the pdf document with evince
	evince ${FILE_TEX}.pdf &

clean:
	# Cleaning ${FILE_TEX} related files...
	ls ${FILE_TEX}.* | grep -v \.tex$ | grep -v \.bib$ | xargs rm -fv
	# Cleaning other tex related files if applicable...
	rm -fv *log *aux *dvi *lof *lot *bit *idx *glo *bbl *ilg *toc *ind *blg *out *nlo *brf *nls *pdf 
	# Cleaning in subdirectories *.aux files...
	find . -regex '.*.aux' -print0 | xargs -0 rm -rfv
	# Cleaning in subdirectories *.log files...
	find . -regex '.*.log' -print0 | xargs -0 rm -rfv
	# Cleaning in subdirectories *.bbl files...
	find . -regex '.*.bbl' -print0 | xargs -0 rm -rfv
	# Cleaning in subdirectories *.blg files...
	find . -regex '.*.blg' -print0 | xargs -0 rm -rfv
	# If there are other generated files, add the previous command again with the proper extension

update:
	# Create a temporal file with the POST request configuration 
	# Uses the authentication parameters of this Makefile
	echo "POST /accounts/ClientLogin HTTP/1.0\nContent-type: application/x-www-form-urlencoded\n\naccountType=${ACCOUNTTYPE}&Email=${EMAIL}&Passwd=${PASSWD}&service=${SERVICE}&source=${SOURCE}" > credentials.txt

	# Perform the authentication
	# Credentials are defined in Makefile
	# and temporally store in updater/credentials.txt
	wget -O clientLogin.txt --no-check-certificate --post-file=credentials.txt "https://www.google.com/accounts/ClientLogin" >/dev/null 2>&1

	# Remove client login information (for security reasons)
	rm credentials.txt
	
	##
	# Get the TEX document
	##

	# Get the document indicated by the first parameter
	wget --header "Authorization: GoogleLogin auth=`cat clientLogin.txt | grep Auth | sed "s#Auth=##" | xargs echo -n`" "https://docs.google.com/feeds/download/documents/Export?docID=${TEX_GOOGLE_DOCS}&exportFormat=txt" -O temp.txt

	# The first line of the downloaded line contains not valid characters
	# Remove first line of the downloaded document
	sed 1d temp.txt > ${FILE_TEX}.tex
	# Remove the temp file
	rm temp.txt

	##
	# Get the BIB document
	##

	# Get the document indicated by the first parameter
	wget --header "Authorization: GoogleLogin auth=`cat clientLogin.txt | grep Auth | sed "s#Auth=##" | xargs echo -n`" "https://docs.google.com/feeds/download/documents/Export?docID=${BIB_GOOGLE_DOCS}&exportFormat=txt" -O temp.txt

	# The first line of the downloaded line contains not valid characters
	# Remove first line of the downloaded document
	sed 1d temp.txt > ${FILE_TEX}.bib
	# Remove the temp file
	rm temp.txt

	# Remove client login information (for security reasons)
	rm clientLogin.txt

edit:
	#Edit main document (Texmaker)
	texmaker ${FILE_TEX}.tex & 