import lxml.html
import beautifulscraper as bs
import xmltodict
import pandas as pd
import numpy as np
import datetime as dt
import _pickle as pk
from tqdm import tqdm
import requests
import feedparser as fp
import glob2 as gl
import psycopg2 as pc

def absee_parser(saveStr):

    fileName = []
    xmlLinks = []
    assetClass = []
    entityType = []
    secName = []
    reportDate = []
    start = 1

    while (fp.parse('https://www.sec.gov/cgi-bin/srch-edgar?text=abs-ee&start='+str(start)+'&count=100&first=2016&last=2017&output=atom').entries != []):

        rssFeed = fp.parse('https://www.sec.gov/cgi-bin/srch-edgar?text=abs-ee&start='+str(start)+'&count=100&first=2016&last=2017&output=atom')
        start += 100

        for i,entry in enumerate(rssFeed.entries):
            connection = bs.urllib2.urlopen(entry.link)
            secName.append(entry.title.replace('ABS-EE - ',''))

            f = entry.title.replace('ABS-EE - ','')+' '+dt.datetime.strptime(entry.updated, '%m/%d/%Y').strftime('%Y%m%d')+' '+entry.summary[55:75]+'.xml'
            print('Scraping data for file: %s ...' %f)
            f = f.replace(' ', '_')
            fileName.append(f)
            reportDate.append(dt.datetime.strptime(entry.updated, '%m/%d/%Y').strftime('%Y%m%d'))

            if ('trust'.lower() in f.lower()):
                entityType.append('Trust')
            else:
                entityType.append('Depositor')

            if ('leas' in f.lower()):
                assetClass.append('Auto Leases')
            elif ('mortgage' in f.lower()) or ('stanley' in f.lower()) or ('bnk4' in f.lower()):
                assetClass.append('CMBS')
            else:
                assetClass.append('Auto Loans')

            dom = lxml.html.fromstring(connection.read())
            for j,l in enumerate(dom.xpath('//a/@href')):  # select the url in href for all a tags(links)
                if ('.xml' in l) and not('103' in l):
                    xmlLinks.append('https://www.sec.gov/'+l)

    dtABS = pd.DataFrame(data=np.column_stack((secName,fileName,entityType,assetClass,reportDate,xmlLinks)),index=np.arange(0,len(fileName),1),
                  columns=['secname','filename','entitytype','assetclass','reportdate','url'])
    dtABS.to_csv('Inputs/fullXMLlist.csv')
    oldABS = []
    for a in np.sort(dtABS['assetclass'].unique()):
        oldABS.append(gl.glob(a+'/*.xml'))
    oldABS = [item for sublist in oldABS for item in sublist]
    oldABS = [o[o.find('/')+1:] for o in oldABS]
    dtABS = dtABS.iloc[np.in1d(dtABS['filename'],oldABS) == False]

    dtABS = dtABS.drop_duplicates(['url','secname'],keep='first')
    dtABS.to_csv('Inputs/'+saveStr,index=False)
    print('Finished scraping ABS postings on SEC Edgar ... ')
    return dtABS

def read_xml_dir(xmlPath):
# Desc: Reads in absurls.csv from the xmlPath.  File contains links to the xml data on the SEC Edgar site.
#xmlPath = 'dtABS.csv'
    dtXml = pd.read_csv('Inputs/'+xmlPath, header=0, index_col=False)
    print('Read in file from path: %s ...' %xmlPath)
    return dtXml

def write_ald_files(dtXml,entityTypes,assetClass):
# Desc: Method writes xml files by pulling data from page specified in URL
# Inputs: dtXml and assets, which is a list of assetTypes e.g. ['Auto Loans','Auto Leases']
# Outputs: Written xml files to a directory prepended with assetType

#entityTypes = ['Trust','Depositor']
#assetClass = ['Auto Loans','Auto Leases']
#e = entityTypes[0]
#a = assetClass[0]

    for e in entityTypes:
        for a in assetClass:
            indx = np.where(np.logical_and(dtXml['entitytype'] == e,dtXml['assetclass'] == a))
            for i in dtXml.iloc[indx].index:
                print('Writing file for securitization: %s' %dtXml['filename'].ix[i])
                response = requests.get(dtXml['url'].ix[i], stream=True)
                with open(dtXml['assetclass'].ix[i] + '/' + dtXml['filename'].ix[i], "wb") as handle:
                    for data in tqdm(response.iter_content()):
                        handle.write(data)

def read_ald_xml(xmlPath):
# Desc: Pulls in xml data from a file and writes it to a pandas dataframe one at a time

    with open(xmlPath) as fd:
        doc = xmltodict.parse(fd.read())

    return pd.DataFrame.from_dict(doc['assetData']['assets'])

def read_ald_files(dtXml,entityType,assetClass):
# Desc: Loops through filenames of files we are interested in loading and creates a pandas dataframe with the data
# Inputs: dtXml and assetType, which is 'Auto Loans' for example
# Outputs: Pandas dataframe with all the merged xml data for that assetType

#entityType = 'Trust'
#assetClass = 'Auto Loans'

    dtOut = pd.DataFrame()
    indx = np.where(np.logical_and(dtXml['entitytype'] == entityType,dtXml['assetclass'] == assetClass))

    for i in dtXml.iloc[indx].index:
        print('Read in file: %s ...' %dtXml['filename'].ix[i])
        dtTemp = read_ald_xml(assetClass + '/' + dtXml['filename'].ix[i])
        dtTemp['securitizationKey'] = dtXml['secname'].ix[i]
        #see if this works
        dtTemp['shelf'] = dtXml['secname'].ix[i][:dtXml['secname'].ix[i].find('2017')-1]
        dtTemp['reportDate'] = dtXml['reportdate'].ix[i]
        dtOut = pd.concat([dtOut,dtTemp],axis=0)

    return dtOut

def write_xml_to_csv(dtXml,entityType,assetClass):
# Desc: Writes xml file data to csv files
# Inputs: dtXml, the field that contains the assetType and the assetType
# Outputs: A series of csv's for the assetType in question
#entityType = 'Trust'
#assetClass = 'Auto Loans'

    indx = np.where(np.logical_and(dtXml['entitytype'] == entityType,dtXml['assetclass'] == assetClass))
    for f in dtXml['filename'].iloc[indx]:
        print('Writing xml file: %s to csv ...' % f)
        read_ald_xml(assetClass + '/' + f).to_csv('csv/' + f + '.csv')

def pickle_save(dtPmts,saveStr):
# Desc: Stores pandas dataframe as a pickled .p file

    pk.dump(dtPmts, open('Pickled/'+saveStr, 'wb'))
    print('Pickled file ...')

def pickle_load(loadStr):
# Desc: Loads object from a pickled .p file

    dtOut = pd.DataFrame()

    for l in loadStr:
        print('Loaded pickled file from: %s...' %l)
        dtOut = pd.concat([dtOut,pk.load(open('Pickled/'+l, 'rb'))],axis=0)

    return dtOut

def import_db_compare():

    try:
        conn = pc.connect(
            "dbname='rawanalyticsdb' user='ungsb1' host='rawanalytics.ch8cnipsctjc.us-east-1.rds.amazonaws.com' password='vWkozJWuDlzdyU6'")
    except:
        print
        "I am unable to connect to the database"

    cur = conn.cursor()
    gg = cur.execute("""SELECT actualinterestcollectedamount FROM rawanalyticsdb_aldloans""")
    rows = cur.fetchall()

def main(argv = sys.argv):

dtXml = absee_parser('dtABS.csv')
dtXml = read_xml_dir('example.csv')
write_ald_files(dtXml,['Trust'],['Auto Loans'])
dtRaw = read_ald_files(dtXml,'Trust','Auto Loans')

pickle_save(dtRaw,'autoLoansTrust20170624partial.p')

dtRaw = pickle_load(['autoLoansTrust20170420.p','autoLoansTrust20170501partial.p','autoLoansTrust20170505partial.p',
                     'autoLoansTrust20170524partial.p','autoLoansTrust20170530partial.p'])
dtRaw = pickle_load(['autoLeasesTrust20170509.p','autoLeasesTrust20170525partial.p','autoLeasesTrust20170526partial.p'])

if __name__ == "__main__":
    sys.exit(main())