from sklearn import ensemble
from sklearn import metrics
import itertools as it
#import seaborn as sns
#import matplotlib.pyplot as plt

class const():
    @staticmethod
    def summaryRowHeader():
        return ['No of Loans', 'Avg Prin Bal $', 'Avg Prin Bal %', 'Avg Loan Size $','BaseR / MSRP %','DBaseR / Sec %',
                'Total TurnIn %','Total Gain/Loss Sale %','Total Def %','Total PP %','Total Delq %','Delq 1m %','Delq 2m %',
                'Delq 3m %','Delq 3m+ %','Def %','PP %','Inc Ver %','Emp Ver %','WAVG APR %',
                'WAVG LTV %', 'WAVG Term m', 'WAVG Age m','WAVG MFC m','WAVG PTI %','WAVG Cons Score',
                'WAVG Comm Score','Cons %','Comm %','Used %', 'New %','Car %','Truck %',
                'SUV %','<2010 %','<2015 %','2015 %','2016 %','2017 %','CA %','TX %',
                'FL %','OH %','NJ %']
    @staticmethod
    def intNo():
        return 7
    @staticmethod
    def rrCols():
        return ['Curr','1m','2m','3m','3m+','CO','PP']
    @staticmethod
    def featTolerance():
        return 10
    @staticmethod
    def maxTrees():
        return 200
    @staticmethod
    def maxDepth():
        return None
    @staticmethod
    def maxLeaf():
        return 1
    @staticmethod
    def modelFields():
        return ['gracePeriodNumber','obligorEmploymentVerificationCode','region','obligorIncomeVerificationLevelCode',
                'originalInterestRatePercentage','originalLoanAmount','originalLoanTerm','paymentToIncomePercentage',
                'remainingTermToMaturityNumber','servicingAdvanceMethodCode','servicingFeePercentage','underwritingIndicator',
                'vehicleManufacturerName','vehicleModelYear','vehicleNewUsedCode','vehicleTypeCode','vehicleValueAmount','vehicleValueSourceCode',
                'consumerCreditScore','commercialCreditScore','age','loanToValueRatio','vintage','primeIndicator']
                # 'modificationTypeCode','subvented','vehicleModelName','originalInterestOnlyTermNumber',,'obligorGeographicLocation'
    @staticmethod
    def frac():
        return .2
    @staticmethod
    def badFields():
        return np.array(['commercialCreditScore','obligorEmploymentVerificationCode','gracePeriodNumber',
                         'obligorIncomeVerificationCode','gracePeriodNumber','servicingFeePercentage',
                         'vehicleManufacturerName'])

def WAVG(dtPmts,fieldStr,weightStr):
# Desc: Creates a weighted average where the averaging field and the weighting fields are specified in a pandas dataframe

    return np.multiply(dtPmts[fieldStr],dtPmts[weightStr]).sum() / dtPmts[weightStr].sum()

def create_summary_row(dtPmts,rowName):

    #rowName = 'Term 0-20'
    if (dtPmts.shape[0] > 0):

        dtRow = pd.DataFrame(data=np.zeros(shape=(1,len(const.summaryRowHeader()))),index=[rowName.title()],
                                 columns=const.summaryRowHeader())

        firstIndx = np.where(dtPmts['monthsFromCutoffDate'] == 0)
        dtRow['Total Def %'].iloc[0] = dtPmts['chargedoffPrincipalAmount'].sum() / dtPmts['reportingPeriodBeginningLoanBalanceAmount'].iloc[firstIndx].sum()
        dtRow['Total PP %'].iloc[0] = dtPmts['principalPrepaid'].sum() / dtPmts['reportingPeriodBeginningLoanBalanceAmount'].iloc[firstIndx].sum()

        dtP = dtPmts.iloc[np.where(dtPmts['summaryDate'] == 1)]

        dtRow['No of Loans'].iloc[0] = dtP.shape[0]
        dtRow['Avg Prin Bal $'].iloc[0] = dtP['reportingPeriodActualEndBalanceAmount'].mean()
        dtRow['Avg Prin Bal %'].iloc[0] = dtP['reportingPeriodActualEndBalanceAmount'].mean() / dtP['originalLoanAmount'].mean()
        dtRow['Avg Loan Size $'].iloc[0] = dtP['originalLoanAmount'].mean()


        dtRow['Total Delq %'].iloc[0] = np.multiply(dtP['currentDelinquencyStatus'] > 0,dtP['reportingPeriodActualEndBalanceAmount']).sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['Delq 1m %'].iloc[0] = np.multiply(dtP['monthsDelinquent'] == 1,dtP['reportingPeriodActualEndBalanceAmount']).sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['Delq 2m %'].iloc[0] = np.multiply(dtP['monthsDelinquent'] == 2,dtP['reportingPeriodActualEndBalanceAmount']).sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['Delq 3m %'].iloc[0] = np.multiply(dtP['monthsDelinquent'] == 3,dtP['reportingPeriodActualEndBalanceAmount']).sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['Delq 3m+ %'].iloc[0] = np.multiply(dtP['monthsDelinquent'] == 4,dtP['reportingPeriodActualEndBalanceAmount']).sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['Def %'].iloc[0] = dtP['chargedoffPrincipalAmount'].sum() / dtP['reportingPeriodActualEndBalanceAmount'].sum()
        dtRow['PP %'].iloc[0] = dtP['principalPrepaid'].sum() / dtP['reportingPeriodBeginningLoanBalanceAmount'].sum()

        dtRow['WAVG LTV %'].iloc[0] = WAVG(dtP,'loanToValueRatio','originalLoanAmount')
        dtRow['WAVG Term m'].iloc[0] = WAVG(dtP,'originalLoanTerm','originalLoanAmount')
        dtRow['WAVG Age m'].iloc[0] = WAVG(dtP,'age','originalLoanAmount')
        dtRow['WAVG MFC m'].iloc[0] = WAVG(dtP,'monthsFromCutoffDate','originalLoanAmount') + 1
        dtRow['WAVG PTI %'].iloc[0] = WAVG(dtP,'paymentToIncomePercentage','originalLoanAmount')

        consIndx = np.where(np.logical_and(dtP['consumerCreditScore'] > 0,np.isnan(dtP['consumerCreditScore']) == False))
        commIndx = np.where(np.logical_and(dtP['commercialCreditScore'] > 0,np.isnan(dtP['commercialCreditScore']) == False))

        if (len(consIndx[0]) > 0):
            dtRow['WAVG Cons Score'].iloc[0] = WAVG(dtP.iloc[consIndx],'consumerCreditScore','originalLoanAmount')
            dtRow['Cons %'].iloc[0] = len(consIndx[0]) / dtP.shape[0]
        if (len(commIndx[0]) > 0):
            dtRow['WAVG Comm Score'].iloc[0] = WAVG(dtP.iloc[commIndx],'commercialCreditScore','originalLoanAmount')
            dtRow['Comm %'].iloc[0] = len(commIndx[0]) / dtP.shape[0]

        dtRow['New %'].iloc[0] = (dtP['vehicleNewUsedCode'] == 1).sum() / dtP.shape[0]
        dtRow['Used %'].iloc[0] = (dtP['vehicleNewUsedCode'] == 2).sum() / dtP.shape[0]

        dtRow['Car %'].iloc[0] = (dtP['vehicleTypeCode'] == 1).sum() / dtP.shape[0]
        dtRow['Truck %'].iloc[0] = (dtP['vehicleTypeCode'] == 2).sum() / dtP.shape[0]
        dtRow['SUV %'].iloc[0] = (dtP['vehicleTypeCode'] == 3).sum() / dtP.shape[0]

        dtRow['<2010 %'].iloc[0] = (dtP['vehicleModelYear'] < 2010).sum() / dtP.shape[0]
        dtRow['<2015 %'].iloc[0] = (dtP['vehicleModelYear'] < 2015).sum() / dtP.shape[0]
        dtRow['2015 %'].iloc[0] = (dtP['vehicleModelYear'] == 2015).sum() / dtP.shape[0]
        dtRow['2016 %'].iloc[0] = (dtP['vehicleModelYear'] == 2016).sum() / dtP.shape[0]
        dtRow['2017 %'].iloc[0] = (dtP['vehicleModelYear'] == 2017).sum() / dtP.shape[0]

        dtRow['CA %'].iloc[0] = (dtP['obligorGeographicLocation'] == 'CA').sum() / dtP.shape[0]
        dtRow['TX %'].iloc[0] = (dtP['obligorGeographicLocation'] == 'TX').sum() / dtP.shape[0]
        dtRow['FL %'].iloc[0] = (dtP['obligorGeographicLocation'] == 'FL').sum() / dtP.shape[0]
        dtRow['NJ %'].iloc[0] = (dtP['obligorGeographicLocation'] == 'NJ').sum() / dtP.shape[0]
        dtRow['OH %'].iloc[0] = (dtP['obligorGeographicLocation'] == 'OH').sum() / dtP.shape[0]

        dtRow['Inc Ver %'].iloc[0] = (dtP['obligorIncomeVerificationLevelCode'] >= 3).sum() / dtP.shape[0]
        dtRow['Emp Ver %'].iloc[0] = (dtP['obligorEmploymentVerificationCode'] >= 3).sum() / dtP.shape[0]

        if ('baseResidualValue' in dtP.columns):
            dtRow['Total TurnIn %'].iloc[0] = np.in1d(dtPmts['terminationIndicator'],[2,4]).sum() / len(dtPmts['assetNumber'].unique())
            dtRow['WAVG APR %'].iloc[0] = WAVG(dtP, 'securitizationDiscountRate', 'originalLoanAmount')
            dtRow['BaseR / MSRP %'].iloc[0] = dtP['baseResidualValue'].sum() / dtP['vehicleValueAmount'].sum()
            dtRow['DBaseR / Sec %'].iloc[0] = np.pv(dtP['securitizationDiscountRate'],dtP['remainingTermToMaturityNumber'],0,
                                                    -dtP['baseResidualValue'],1).sum() / dtP['reportingPeriodBeginningLoanBalanceAmount'].sum()
            gainIndx = np.where(np.abs(dtPmts['saleGainOrLoss']) > 0)
            if (len(gainIndx[0]) > 0):
                dtRow['Total Gain/Loss Sale %'].iloc[0] = dtPmts['saleGainOrLoss'].iloc[gainIndx].sum() / dtPmts['contractResidualValue'].iloc[gainIndx].sum()
        else:
            dtRow['WAVG APR %'].iloc[0] = WAVG(dtP, 'originalInterestRatePercentage', 'originalLoanAmount')
            dtRow = dtRow.drop(['BaseR / MSRP %','DBaseR / Sec %','Total TurnIn %','Total Gain/Loss Sale %'],axis=1)

        return dtRow

    else:
        print('Warning: No rows in segmentation ...')

def create_summary_strats(dtPmts,functionStr,fieldStr,rowStr,labels,factor):

    dtSumm = create_summary_row(dtPmts, rowStr + ': Overall')
    if (functionStr == 'stratify'):
    # Case 1: Where the program stratifies into 7 equally spaced categories
        intLength = (dtPmts[fieldStr].max() - dtPmts[fieldStr].min()) / const.intNo()
        startNo = float(dtPmts[fieldStr].min()) + np.arange(0, const.intNo(),1) * intLength
        endNo = float(dtPmts[fieldStr].min()) + np.arange(1, const.intNo() + 1,1) * intLength
        for i in range(0, const.intNo()):
            rowIndx = np.where(np.logical_and.reduce([dtPmts[fieldStr] >= startNo[i],dtPmts[fieldStr] < endNo[i],np.isnan(dtPmts[fieldStr]) == False]))
            dtSumm = pd.concat([dtSumm,create_summary_row(dtPmts.iloc[rowIndx],
                                    rowStr + ': ' + str("{0:.2f}".format(startNo[i])) + ' - ' + str("{0:.2f}".format(endNo[i])))],axis=0)

    elif (functionStr == 'labels'):
    # Case 2: Where categories are specified ahead of time
        for i in range(0, len(labels)):
            rowIndx = np.where(dtPmts[fieldStr] == i + factor)
            dtSumm = pd.concat([dtSumm, create_summary_row(dtPmts.iloc[rowIndx], rowStr + ': ' + labels[i])], axis=0)

    elif (functionStr == 'unique'):
    # Case 3: Where categories are determined by unique values of a text column
        labels = np.sort(dtPmts[fieldStr].unique())
        for i in range(0, len(labels)):
            rowIndx = np.where(dtPmts[fieldStr] == labels[i])
            dtSumm = pd.concat([dtSumm, create_summary_row(dtPmts.iloc[rowIndx], rowStr + ': ' + str(labels[i]))], axis=0)

    print('Calculating row: %s ...' %rowStr)

    return dtSumm

def create_summary(dtPmts):

    print('Creating summary report ...')
    return pd.concat([create_summary_strats(dtPmts,'stratify','originalLoanTerm','Term',[],[]),
                        create_summary_strats(dtPmts,'stratify','originalLoanAmount','Size',[],[]),
                        create_summary_strats(dtPmts,'stratify','nextInterestRatePercentage','APR',[],[]),
                        create_summary_strats(dtPmts,'stratify','age','Age',[],[]),
                        create_summary_strats(dtPmts,'stratify','consumerCreditScore','FICO',[],[]),
                        create_summary_strats(dtPmts,'stratify','commercialCreditScore','CommScore',[],[]),
                        create_summary_strats(dtPmts,'stratify','loanToValueRatio','LTV',[],[]),
                        create_summary_strats(dtPmts,'stratify','paymentToIncomePercentage','PTI',[],[]),
                        create_summary_strats(dtPmts,'labels','vehicleTypeCode','VehType',['Car', 'Truck', 'SUV'],1),
                        create_summary_strats(dtPmts,'labels','vehicleNewUsedCode','VehCond',['New', 'Used'],1),
                        create_summary_strats(dtPmts,'labels','vehicleModelYear', 'VehYear', ['2012','2013','2014','2015','2016','2017'],2012),
                        create_summary_strats(dtPmts,'unique','obligorEmploymentVerificationCode','EmpVer',[],[]),
                        create_summary_strats(dtPmts,'unique','obligorIncomeVerificationLevelCode','IncVer',[],[]),
                        create_summary_strats(dtPmts, 'unique', 'vehicleManufacturerName', 'VehManu', [], [])],axis=0)

def create_comparison(dtPmts):

    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    dtComp = create_summary_row(dtPmts,'All Trusts')

    for s in secKeys:
        secIndx = np.where(dtPmts['securitizationKey'] == s)
        dtComp = pd.concat([dtComp,create_summary_row(dtPmts.iloc[secIndx],s)],axis=0)
        print('Calculating strats for securitization: %s ...' %s)

    return dtComp

def create_performance(dtPmts,axisStr,assetStr):
# Plots specific outputs based on method
#axisStr = 'monthsFromCutoffDate'

    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    secKeys = np.append(secKeys,'All Trusts')

    ABSTable = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' ABSspeed')
    ppTable = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' Prepays')
    delq30Table = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' Delq30')
    delq60Table = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' Delq60')
    delq90Table = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' Delq90')
    defTable = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' Def')
    CNLTable = pd.DataFrame(data=np.ones(shape=(dtPmts[axisStr].unique().shape[0],len(secKeys)))*np.nan,
                            index=np.sort(dtPmts[axisStr].unique()),columns=secKeys+' CNL')

    for s in secKeys:
        if ('Trust' in s):

            if (s == 'All Trusts'):
                secIndx = np.arange(0,dtPmts.shape[0])
            else:
                secIndx = np.where(dtPmts['securitizationKey'] == s)

            if (assetStr == 'Auto Loans'):
                dtSMM = dtPmts.iloc[secIndx].groupby([axisStr])['reportingPeriodBeginningLoanBalanceAmount','chargedoffPrincipalAmount','reportingPeriodActualEndBalanceAmount',
                                                            'actualPrincipalCollectedAmount','otherPrincipalAdjustmentAmount','scheduledPrincipalAmount'].sum()
                SMMTable = (dtSMM['actualPrincipalCollectedAmount'] + dtSMM['otherPrincipalAdjustmentAmount'] + dtSMM['chargedoffPrincipalAmount'] - dtSMM['scheduledPrincipalAmount']) / \
                                                        (dtSMM['reportingPeriodBeginningLoanBalanceAmount'] - dtSMM['scheduledPrincipalAmount'])
                ABSTable[s+' ABSspeed'] = SMMTable/(1-SMMTable*(WAVG(dtPmts.iloc[secIndx],'ageFromCutoffDate','beginningBalanceAtCutoffDate')-1)).values
            elif (assetStr == 'Auto Leases'):
                dtSMM = dtPmts.iloc[secIndx].groupby([axisStr])['reportingPeriodBeginningLoanBalanceAmount','reportingPeriodActualEndBalanceAmount','ageFromCutoffDate','beginningBalanceAtCutoffDate',
                                                                'acquisitionCost','scheduledSecuritizationBeginValueAmount','scheduledSecuritizationEndValueAmount'].sum()

                dtSurv = 1 - (dtSMM['reportingPeriodBeginningLoanBalanceAmount'] / dtSMM['scheduledSecuritizationBeginValueAmount']) / (dtSMM['reportingPeriodActualEndBalanceAmount'] / dtSMM['scheduledSecuritizationEndValueAmount'])
                ABSTable[s +' ABSspeed'] = dtSurv / (1 + dtSurv * WAVG(dtPmts.iloc[secIndx],'ageFromCutoffDate','beginningBalanceAtCutoffDate')).values

            ppTable[s+' Prepays'] = dtPmts.iloc[secIndx].groupby([axisStr])['principalPrepaid'].sum() / dtPmts.iloc[secIndx].groupby([axisStr])['reportingPeriodBeginningLoanBalanceAmount'].sum()

            delqTable = pd.pivot_table(data=dtPmts.iloc[secIndx],values='reportingPeriodActualEndBalanceAmount',index=axisStr,columns='monthsDelinquent',aggfunc=np.sum)

            if (1 in delqTable.columns):
                delq30Table[s+' Delq30'] = delqTable[1]/delqTable.sum(axis=1)
            if (2 in delqTable.columns):
                delq60Table[s+' Delq60'] = delqTable[2]/delqTable.sum(axis=1)
            if (3 in delqTable.columns):
                delq90Table[s+' Delq90'] = delqTable[3]/delqTable.sum(axis=1)
            if (5 in delqTable.columns):
                defTable[s+' Def'] = delqTable[5]/delqTable.sum(axis=1)

            if (axisStr == 'monthsFromCutoffDate'):
                CNLstr = 'beginningBalanceAtCutoffDate'
                CNLdivisor = dtPmts[[CNLstr, axisStr]].iloc[secIndx]
            else:
                CNLstr = 'originalLoanAmount'
                CNLdivisor = dtPmts[[CNLstr, axisStr, 'monthsFromCutoffDate']].iloc[secIndx]

            CNLdivisor = dtPmts[CNLstr].iloc[np.where(CNLdivisor['monthsFromCutoffDate'] == 0)].sum()
            CNLTable[s+' CNL'] = (dtPmts.iloc[secIndx].groupby([axisStr])['netLosses'].sum() / CNLdivisor).cumsum()

            print('Calculated performance for securitization: %s on axis: %s ...' %(s.title(),axisStr))

    outTable = pd.concat([ABSTable,defTable,delq60Table,CNLTable],axis=1)

    return outTable

def plot_performance(dtPmts):

    plt.rcParams.update({'font.size': 8})
    f, axArr = plt.subplots(3, 6)
    f.suptitle('ALD Performance Charts')

    axisStr = ['monthsFromCutoffDate','age','reportingPeriodBeginningDate']
    cols = ['ABSspeed','Prepays','Delq30','Delq60','Delq90','CNL']

    for i,axis in enumerate(axisStr):
        dtPerf = create_performance(dtPmts,axis)
        for j,c in enumerate(cols):

            print(i,j)
            colIndx = [c in col for col in dtPerf.columns]
            axArr[i, j].set_xlabel(axis)
            if (j == 0) & (i == 0):
                dtPerf.loc[:,colIndx].plot(kind='line',ax=axArr[i,j],subplots=False,layout=(i,j),legend=True)
                axArr[i,j].legend(bbox_to_anchor=(.75, -2.62),loc=2,borderaxespad=0.,ncol=4,prop={'size':6})
            else:
                dtPerf.loc[:,colIndx].plot(kind='line', ax=axArr[i, j],subplots=False,layout=(i, j),legend=False)

            if (i==0):
                axArr[i, j].set_title(c)

            #if (axis == 'reportingPeriodBeginningDate'):
            #    plt.sca(axArr[i,j])
            #    newTicks = [dt.datetime.toordinal(d) for d in dtPerf.index]
            #    plt.xticks(newTicks,dtPerf.index.strftime('%b%y'))

def create_curves(dtPmts,axisStr):
#axisStr = 'age'
#axisStr = ['monthsFromCutoffDate','age','reportingPeriodBeginningDate']
    fieldStr = ['Curr','1mDelq','2mDelq','3mDelq','3m+Delq','Def','PP']
    primeStates = ['Other','Subprime <640','Nearprime 640-680','Prime 680-740','Superprime >740']
    dtCurves = pd.DataFrame(np.zeros(shape=(len(dtPmts[axisStr].unique()),len(fieldStr)*len(primeStates))),
                            columns=np.arange(0,len(fieldStr)*len(primeStates),1))
    cols=[]

    for p,prime in enumerate(primeStates):

        print('Creating curve for prime category: %s ...' %prime)
        dtBal = pd.pivot_table(data=dtPmts.iloc[np.where(dtPmts['primeIndicator']==p)],
                               values='reportingPeriodBeginningLoanBalanceAmount',columns='monthsDelinquent',index=axisStr, aggfunc=np.sum).fillna(0).cumsum()
        dtBal = dtBal.div(dtPmts['originalLoanAmount'].iloc[np.where(dtPmts['primeIndicator'] == p)].sum(),axis=0).fillna(method='ffill')
        dtCurves[dtBal.columns.values + len(fieldStr) * p] = dtBal

        cols += [prime + ' ' + field for field in fieldStr]

    dtCurves.columns = cols
    dtCurves = dtCurves.fillna(method='ffill')

    # Reorder columns by name
    cols = []
    for s in ['PP','Def','1mDelq','2mDelq','3mDelq','3m+Delq']:
        cols = np.append(cols,[c for c in dtCurves.columns if s in c ])

    return dtCurves[cols]

def create_rollrates_matrix(dtPmts):

    dtMatrix = np.zeros(shape=(len(const.rrCols()), len(const.rrCols())))

    for stState in np.sort(dtPmts['monthsDelinquent'].unique()):
        for endState in np.sort(dtPmts['monthsDelinquent'].unique()):
            #print(stState,endState)
            preIndx = np.where(np.logical_and(dtPmts['monthsDelinquent'] == stState, dtPmts['summaryDate'] == 0))
            postIndx = preIndx[0][np.where(dtPmts['monthsDelinquent'].iloc[preIndx[0] + 1] == endState)]
            if (dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[preIndx].sum() != 0):
                dtMatrix[stState,endState] = dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[postIndx].sum() / dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[preIndx].sum()

    return pd.DataFrame(data=dtMatrix,index=const.rrCols(),columns=const.rrCols())

def create_rollrates_ts(dtPmts,axisStr):

#axisStr = 'age'
    dtOut = np.zeros(shape=(dtPmts[axisStr].unique()[:-1].shape[0],(len(const.rrCols())-2)*len(const.rrCols())))
    m = 0
    cols = list()

    for s in range(0,len(const.rrCols())-2):
        for e in range(0,len(const.rrCols())):
            cols.append(const.rrCols()[s] + '-' + const.rrCols()[e])
            for i,ind in enumerate(np.sort(dtPmts[axisStr].unique()[:-1])):
                print('Calculating where stState: %s, endState: %s, and axis: %s ... ' %(const.rrCols()[s],const.rrCols()[e],ind))
                preIndx = np.where(np.logical_and.reduce([dtPmts[axisStr] == ind,
                                                          dtPmts['monthsDelinquent'] == s,
                                                          dtPmts['summaryDate'] == 0]))
                postIndx = preIndx[0][np.where(dtPmts['monthsDelinquent'].iloc[preIndx[0]+1] == e)]
                if (dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[preIndx].sum() != 0):
                    dtOut[i,m] = dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[postIndx].sum() / dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[preIndx].sum()
            m += 1

    return pd.DataFrame(data=dtOut,index=np.sort(dtPmts[axisStr].unique()[:-1]),columns=cols)

def plot_model_outputs(auc,oob,rocY,rocX,nTrees,headers,featImp,superTitle):

    #Plots specific outputs based on method

    plt.rcParams.update({'font.size':8})
    f, axArr = plt.subplots(2, 2)
    f.suptitle(superTitle)

    # plot ROC curves
    ctClass = [i * 0.01 for i in range(0,101,1)]
    axArr[0,0].plot(ctClass, ctClass, label='x=y',linestyle=':')
    for i in range(0,rocY.shape[0]):
        axArr[0,0].plot(rocX, rocY[i,:], label='ROC Curve for iTrees:'+str(nTrees[i]),linewidth=2)

    axArr[0,0].set_xlabel('False Positive Rate')
    axArr[0,0].set_ylabel('True Positive Rate')
    axArr[0,0].set_title('ROCs vs No. of Trees')
    #axArr[0,0].legend(loc='upper center',shadow=False,)

    # plot avg top 10 features by importance
    axArr[0,1].barh(np.arange(const.featTolerance())+.5,featImp[0:const.featTolerance()],align='center')
    axArr[0,1].set_yticks(np.arange(const.featTolerance())+.5)
    axArr[0,1].set_yticklabels(headers[0:const.featTolerance()])
    axArr[0,1].set_title('Var Importance for No. of Trees Range Mid')

    # plot AUC vs No. of Trees
    axArr[1,0].plot(list(nTrees),auc)
    axArr[1,0].set_xlabel('No. of Trees')
    axArr[1,0].set_ylabel('AUC')
    axArr[1,0].set_title('AUC vs No. of Trees')

    # plot OOB Errors vs No. of Trees
    axArr[1,1].plot(list(nTrees),oob)
    axArr[1,1].set_xlabel('No. of Trees')
    axArr[1,1].set_ylabel('OOB Error')
    axArr[1,1].set_title('OOB Error (1-OOB Score) vs No. of Trees')

def generate_regressors(dtR):
    # Converts input data to classification columns and I tried to drop the columns containing 20% of the non-missing values
#dtR = dtX
    cols = []
    dtOut = np.zeros(shape=(dtR.shape[0],0))
    sparseCount = dtR.count(axis=0)/dtR.shape[0] # percentage of non-nans in columns

    topTolerance = .9
    bottomTolerance = .1
    freqTolerance = .15
    sparseTolerance = .2
    binTolerance = 10
    bins = range(0,int((1-sparseTolerance)*100),binTolerance)

    for i in range(0,dtR.shape[1]):

        #print(dtR.columns[i],i)
        freqTable = dtR.iloc[:,i].value_counts()/dtR.iloc[:,i].count()

        if (freqTable.shape[0] == 0):
        # Case 0: Column is all nans or blanks
            print('Case 0, for i = %d ...' %(i))

        elif (freqTable.iloc[0] >= topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 1: If most frequent value is > 90% and column is not sparse, then (is frequent value, is missing)
            dtTemp = np.zeros(shape=(dtR.shape[0],1))
            cols.append(dtR.columns.values[i] + "_is: " + str(dtR.iloc[:, i].value_counts().index[0]))
            dtTemp[np.where(dtR.iloc[:, i] == dtR.iloc[:, i].value_counts().index[0]),0] = 1
            dtOut = np.concatenate((dtOut, dtTemp), axis=1)
            print('Case 1, for i = %d ...' %(i))

        elif (bottomTolerance <= freqTable.iloc[0] < topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 2: If most frequent value is in between tolerances and column is not sparse, then (is multiple vals above threshold, is missing)

            uniqCols = freqTable.index[(freqTable > freqTolerance).nonzero()[0]]
            if uniqCols.shape[0] == freqTable.shape[0]:
            # if uniqCols contains all values then drop last one
                uniqCols = uniqCols[0:uniqCols.shape[0]-1]

            dtTemp = np.zeros(shape=(dtR.shape[0],uniqCols.shape[0]))
            # pulls values for columns

            for j,u in enumerate(uniqCols): # parse possible values
                cols.append(dtR.columns.values[i] + "_is: " + str(u))
                dtTemp[np.where(dtR.iloc[:,i] == u),j] = 1
                #print '\ni = %d, j = %d' % (i, j)

            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            print('Case 2, for i = %d ...' %(i))

        elif (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 3: Else numbers are numerical
            dtTemp = np.zeros(shape=(dtR.shape[0],len(bins)))
            # divide column into deciles and exclude last 20%
            for j,u in enumerate(bins):
                print(j,u)
                dtTemp[np.where(np.logical_and(dtR.iloc[:,i]>=np.percentile(dtR.iloc[:,i],u),dtR.iloc[:,i]<np.percentile(dtR.iloc[:,i],u+binTolerance))),j] = 1
                cols.append(dtR.columns.values[i] + "_bin: " + str(u))

            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            print('Case 3, for i = %d ...' %(i))

        else:
        # Case 4: Values are either one value or numbers are too sparse - turns out only one column falls into this case, which was dropped
            print('Case 4, for i = %d ...' %(i))

        if (dtR.iloc[:,i].isnull().sum() > 0) & (dtR.iloc[:,i].isnull().sum() < dtR.shape[0]):
        # Case 4: Add a column if some values are missing or if there is 1 value and blanks
            dtTemp = np.zeros(shape=(dtR.shape[0],1))
            dtTemp[np.where(dtR.iloc[:,i].isnull()),0] = 1
            cols.append(dtR.columns.values[i] + "_is: null")
            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            print('Adding col for missing vals, for i = %d ...' %(i))

    print('\nFinished processing input data...')
    #temp = pd.DataFrame(data=dtOut, index=dtR.index, columns=cols)
    #temp.to_csv('temp.csv')
    return pd.DataFrame(data=dtOut,index=dtR.index,columns=cols)

def create_heatmap(dtPmts,trustStr):

#trustStr = 'Ally Auto Receivables Trust 2017-2'

    if (trustStr == 'All'):
        dtX = dtPmts[const.modelFields()]
        dtX = dtX.sample(frac=const.frac(),replace=True)
    else:
        trustIndx = np.where(dtPmts['securitizationKey'] == trustStr)
        dtX = dtPmts[const.modelFields()].iloc[trustIndx]

    Y = np.ravel(np.logical_and(dtPmts['monthsDelinquent'].loc[dtX.index] > 1,dtPmts['monthsDelinquent'].loc[dtX.index] < 6))
    dtX = generate_regressors(dtX)
    #dtX.to_csv('dtX.csv')

    nTrees = range(100,110,10)
    auc = []
    oob = []
    rocX = np.arange(0, 1.01, .01)
    rocY = np.zeros(shape=(len(nTrees), rocX.shape[0]))

    for i,trees in enumerate(nTrees):
        print('\nUsing Bagging + Random Forests generating trial:%d with trees:%d ...' % (i, trees))

        rfModel = ensemble.RandomForestClassifier(n_estimators=trees, max_depth=const.maxDepth(),max_features='auto',
                                            bootstrap=True,oob_score=True,random_state=531,min_samples_leaf=const.maxLeaf())
        rfModel.fit(dtX,Y)
        predVector = rfModel.predict_proba(dtX)

        # generate ROC, AUC, OOB, and feature importance diagnostic stats
        if (predVector.shape[1] > 1):

            fpr, tpr, thresh = metrics.roc_curve(Y, predVector[:,1])
            rocY[i,:] = np.interp(rocX,fpr,tpr)

            auc.append(metrics.roc_auc_score(Y, predVector[:, 1]))
            oob.append(1-rfModel.oob_score_)

        else:
            print('No model because there are no differences in the Y values in the test set ...')

    #plot_model_outputs(auc,oob,rocY,rocX,nTrees,dtX.columns.values[featImpIndx],featImp[featImpIndx],'Bagging + Random Forest Model')

    featImp = rfModel.feature_importances_ / rfModel.feature_importances_.max()
    featImpIndx = np.argsort(featImp)[::-1]

    return pd.DataFrame(data=np.concatenate((np.expand_dims(dtX.columns.values[featImpIndx], axis=1),
                                             np.expand_dims(featImp[featImpIndx], axis=1)),axis=1),
                                            columns=['colName', 'colImp'])

def get_heatmap_cols(dtCol):
#dtCol = dtPmts['loanToValueRatio']
    dtHist = np.zeros(shape=(dtCol.shape[0],1))

    if (np.issubdtype(dtCol.dtype, np.number)) & (dtCol.unique().shape[0] > const.intNo()):
        intLength = (dtCol.max() - dtCol.min()) / const.intNo()
        startNo = float(dtCol.min()) + np.arange(0, const.intNo(), 1) * intLength
        endNo = float(dtCol.min()) + np.arange(1, const.intNo() + 1, 1) * intLength
        categoryList = [str("{0:.1f}".format(startNo[i]))+'-'+str("{0:.1f}".format(endNo[i])) for i in range(0,const.intNo())]
        for c in range(0, const.intNo()):
            rowIndx = np.where(np.logical_and.reduce([dtCol>=startNo[c],dtCol< endNo[c],np.isnan(dtCol) == False]))
            dtHist[rowIndx,0] = c
    else:
        categoryList = np.sort(dtCol.unique())
        for c in range(0,len(categoryList)):
            #print(c)
            rowIndx = np.where(dtCol == categoryList[c])
            dtHist[rowIndx,0] = c

    return dtHist,np.array(categoryList)

def plot_heatmaps(dtPmts,dtImp,trustStr):
#trustStr = 'Santander Drive Auto Receivables Trust 2017-1'

    dtTrust = dtPmts.iloc[np.where(dtPmts['securitizationKey'] == trustStr)]

    plt.rcParams.update({'font.size': 6})
    f, axArr = plt.subplots(2, 3)
    f.suptitle('Important Features Heatmaps for ' + trustStr + ' (Z = Mean of Def + Delq > 1m)')

    dtImp['fieldname'] = np.array([f[0:f.find('_',0,len(f))] for f in dtImp['colName'].values])
    dropIndx = np.where(np.in1d(dtImp['fieldname'],const.badFields()))
    dtImp = dtImp.drop(dtImp.index[dropIndx],axis=0)
    fieldsList = dtImp['fieldname'].drop_duplicates(keep='first').iloc[:4].values
    impactFields = fieldsList

    fieldsList = list(it.combinations(fieldsList,2))
    delqIndx = np.where(np.logical_and(dtTrust['monthsDelinquent'] > 1,dtTrust['monthsDelinquent'] < 6))

    for fieldsIndx,fields in enumerate(fieldsList):

        print('Creating heatmap for fields: %s ... '%str(fields))
        i = fieldsIndx//3
        j = fieldsIndx%3

        dtHeatmap = pd.DataFrame(data=np.zeros(shape=(dtTrust.shape[0], 3)),index=dtTrust.index,columns=[fields[0],fields[1],'BadHombres'])
        dtHeatmap[fields[0]], xCat = get_heatmap_cols(dtTrust[fields[0]])
        dtHeatmap[fields[1]], yCat = get_heatmap_cols(dtTrust[fields[1]])
        dtHeatmap['BadHombres'].iloc[delqIndx] = 1

        dtHeatmap = pd.pivot_table(data=dtHeatmap,index=fields[0],columns=fields[1],values='BadHombres')
        dtHeatmap.index = xCat[dtHeatmap.index.values.astype(int)]
        dtHeatmap.columns = yCat[dtHeatmap.columns.values.astype(int)]

        sns.set(font_scale=.75)
        sns.heatmap(data=dtHeatmap,xticklabels=True,yticklabels=True,ax=axArr[fieldsIndx//3,fieldsIndx%3],annot=True,annot_kws={"size":6},cbar=False)
        axArr[i,j].set_xlabel(fields[1],fontsize=7)
        axArr[i,j].set_ylabel(fields[0],fontsize=7)
        axArr[i,j].set_xticklabels(axArr[i,j].get_xticklabels(),rotation=25, fontsize=6)
        axArr[i,j].set_yticklabels(axArr[i,j].get_yticklabels(),rotation=45, fontsize=6)

    plt.savefig('Heatmaps/'+ trustStr + '.png',dpi=200)
    plt.close()

    return pd.DataFrame(data=impactFields,columns=[trustStr],index=np.arange(0,impactFields.shape[0],1))

def run_heatmaps(dtPmts):

    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    #s = 'Santander Drive Auto Receivables Trust 2017-1'
    dtFields = pd.DataFrame()

    for s in secKeys:
        print(s)
        dtImp = create_heatmap(dtPmts,s)
        if (dtImp['colImp'].isnull().all() == False):
            dtFields = pd.concat([dtFields,plot_heatmaps(dtPmts,dtImp,s)],axis=1)

    return dtFields

def describe_data(dtPmts):

    dtPmts = dtPmts.drop(['subvented', 'modificationTypeCode'], axis=1)
    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    dtDesc = pd.DataFrame()

    for s in secKeys:
        print(s)
        secIndx = np.where(dtPmts['securitizationKey']==s)
        dtTemp = dtPmts.iloc[secIndx].describe(percentiles=[.25, .5, .75], exclude=[np.number])
        dtTemp.index = s + ' ' + dtTemp.index
        dtDesc = pd.concat([dtDesc,dtTemp],axis=0)

    dtDesc.to_csv('dtDesc.csv')

def main(argv = sys.argv):

create_summary(dtPmts).to_csv('dtSumm.csv')
create_comparison(dtPmts).to_csv('dtComp.csv')
pd.concat([create_performance(dtPmts,'monthsFromCutoffDate','Auto Loans'),create_performance(dtPmts,'reportingPeriodBeginningDate','Auto Loans'),create_performance(dtPmts,'age','Auto Loans')],axis=0).to_csv('dtPerf.csv')
create_performance(dtPmts,'monthsFromCutoffDate','Auto Leases')
plot_performance(dtPmts)

create_curves(dtPmts,'age').to_csv('dtc.csv')
create_rollrates_ts(dtPmts,'age').to_csv('rrts.csv')
create_rollrates_matrix(dtPmts).to_csv('rrmat.csv')

dtFields = run_heatmaps(dtPmts)

dtImp = create_heatmap(dtPmts,'GM Financial Automobile Leasing Trust 2017-1')
plot_heatmaps(dtPmts,dtImp,'GM Financial Automobile Leasing Trust 2017-1')

ii = np.where(dtRaw['vehicleManufacturerName'] == 'MERC')
dtRaw.iloc[ii].to_csv('merc.csv')

if __name__ == "__main__":
    sys.exit(main())