import pandas as pd
import numpy as np

class const():
    @staticmethod
    def booleanFields():
        return ['assetAddedIndicator','assetSubjectDemandIndicator','coObligorIndicator','reportingPeriodModificationIndicator',
                'repossessedIndicator','underwritingIndicator']
    @staticmethod
    def dateFields():
        return ['originalFirstPaymentDate','originationDate','reportingPeriodBeginningDate','reportingPeriodEndingDate',
                'zeroBalanceEffectiveDate']
    @staticmethod
    def debugFieldsClean():
        return ['reportingPeriodBeginningDate','assetNumber','securitizationKey','reportingPeriodBeginningLoanBalanceAmount',
                'actualPrincipalCollectedAmount','chargedoffPrincipalAmount','otherPrincipalAdjustmentAmount','reportingPeriodActualEndBalanceAmount',
                'actualInterestCollectedAmount','recoveredAmount','repossessedProceedsAmount','principalPrepaid','dueAmount',
                'scheduledPrincipalAmount','nextReportingPeriodPaymentAmountDue','currentDelinquencyStatus','totalActualAmountPaid']
    @staticmethod
    def debugFieldsRaw():
        return ['reportingPeriodBeginningDate', 'assetNumber', 'securitizationKey','reportingPeriodBeginningLoanBalanceAmount',
                'actualPrincipalCollectedAmount', 'chargedoffPrincipalAmount', 'otherPrincipalAdjustmentAmount',
                'reportingPeriodActualEndBalanceAmount','actualInterestCollectedAmount', 'recoveredAmount',
                'repossessedProceedsAmount','scheduledPrincipalAmount','nextReportingPeriodPaymentAmountDue','currentDelinquencyStatus',
                'totalActualAmountPaid']
    @staticmethod
    def decimalFields():
        return ['actualInterestCollectedAmount','actualOtherCollectedAmount','actualPrincipalCollectedAmount','chargedoffPrincipalAmount',
                'originalLoanAmount','otherAssessedUncollectedServicerFeeAmount','otherPrincipalAdjustmentAmount',
                'recoveredAmount','reportingPeriodActualEndBalanceAmount','reportingPeriodBeginningLoanBalanceAmount',
                'reportingPeriodScheduledPaymentAmount','totalActualAmountPaid',
                'nextReportingPeriodPaymentAmountDue', 'repossessedProceedsAmount', 'scheduledInterestAmount',
                'scheduledPrincipalAmount','vehicleValueAmount','servicerAdvancedAmount','servicingFlatFeeAmount','otherServicerFeeRetainedByServicer',]
    @staticmethod
    def integerFields():
        return ['currentDelinquencyStatus','gracePeriodNumber','interestCalculationTypeCode',
                'obligorCreditScore','obligorEmploymentVerificationCode','obligorIncomeVerificationLevelCode',
                'originalInterestRateTypeCode','originalLoanTerm','paymentExtendedNumber',
                'paymentTypeCode','remainingTermToMaturityNumber','servicingAdvanceMethodCode','vehicleModelYear','vehicleNewUsedCode',
                'vehicleTypeCode','vehicleValueSourceCode','zeroBalanceCode','originalInterestOnlyTermNumber']
    @staticmethod
    def listFields():
        return ['modificationTypeCode','subvented']
    @staticmethod
    def rateFields():
        return ['nextInterestRatePercentage','originalInterestRatePercentage',
                'paymentToIncomePercentage','reportingPeriodInterestRatePercentage','servicingFeePercentage']
    @staticmethod
    def stringFields():
        return ['assetNumber','assetTypeNumber','obligorCreditScoreType','obligorGeographicLocation','originatorName','primaryLoanServicerName',
                'securitizationKey','vehicleManufacturerName','vehicleModelName']
    @staticmethod
    def rawCols():
        return ['Count','OpenBal','StartMonth','EndMonth','MissingMonths','Walk','IncrBal','Pmts','Missing','Extra',
                'COExtra','Dupes','NegOpenBal','NegCloseBal','RateNeg','RatePos','Integer','NegCO','PartialCO','GreaterCO','NegRepo','NegRecov']
    @staticmethod
    def minSens():
        return .001
    @staticmethod
    def divInd():
        return .4
    @staticmethod
    def divMax():
        return 5

def read_dict(path):
# Desc: Reads in dict from the path.  File contains lookups to append to dataframe.

    dtD = pd.read_csv('Inputs/'+path, header=0, index_col=False)
    print('Read in file from path: %s ...' %path)
    return dtD

def clean_ald_files(dtPmts):
# Desc: Cleans raw ald file data and puts it into a format that is usable for reporting
# Inputs: The raw ALD payment dataframe
# Outputs: The cleaned ALD payment dataframe

    # dtPmts['shelf'] = dtPmts['securitizationKey']
    # secKeys = np.sort(dtPmts['securitizationKey'].unique())
    # for s in secKeys:
    #     secIndx = np.where(dtPmts['securitizationKey'] == s)
    #     dtPmts['shelf'].iloc[secIndx] = s[:s.find('2017') - 1]
    # print('Added shelf ... ')

    # Convert date to datetime objects
    dtPmts['reportingPeriodBeginningDate'] = pd.to_datetime(dtPmts['reportingPeriodBeginningDate'], format='%m-%d-%Y')
    dtPmts['reportingPeriodEndingDate'] = pd.to_datetime(dtPmts['reportingPeriodEndingDate'], format='%m-%d-%Y')
    dtPmts['originationDate'] = pd.to_datetime(dtPmts['originationDate'], format='%m/%Y')
    dtPmts['originalFirstPaymentDate'] = pd.to_datetime(dtPmts['originalFirstPaymentDate'], format='%m/%Y')
    dtPmts['zeroBalanceEffectiveDate'] = pd.to_datetime(dtPmts['zeroBalanceEffectiveDate'], format='%m/%Y')

    print('Cleaned dates ...')

    # Replace strings with numbers (No score = NaN)
    replaceStr = const.decimalFields() + const.integerFields() + const.rateFields()

    for r in replaceStr:
        dtPmts[r] = pd.to_numeric(dtPmts[r], errors='coerce')
        print('Cleaned field: %s ...' % r)

    # Replace nans on certain fields to 0
    nanStr = ['chargedoffPrincipalAmount','obligorIncomeVerificationLevelCode','obligorEmploymentVerificationCode',
              'obligorCreditScore','repossessedProceedsAmount','recoveredAmount','otherPrincipalAdjustmentAmount',
              'currentDelinquencyStatus']
    for n in nanStr:
        dtPmts[n] = dtPmts[n].fillna(0)
    print('Cleaned nans ...')

    # Drop duplicates - Carmax u vs b
    dtPmts = dtPmts.drop_duplicates(subset=['assetNumber','reportingPeriodBeginningDate','securitizationKey'],keep='last')
    #dtPmts = dtPmts.sort_values(by=['assetNumber','reportingPeriodBeginningDate','securitizationKey'],axis=0)
    #dtPmts = dtPmts.reset_index(drop=True)
    print('Dropped duplicate records and reindexed ...')

    # Fix manufacturer names
    dtManus = read_dict('manus.csv')
    dtPmts = pd.merge(dtPmts, dtManus, how='left', left_on='vehicleManufacturerName', right_on='old', copy=False)
    dtPmts = dtPmts.drop(['old', 'vehicleManufacturerName'], axis=1)
    dtPmts = dtPmts.rename(columns={'new': 'vehicleManufacturerName'})
    dtPmts['vehicleManufacturerName'] = dtPmts['vehicleManufacturerName'].fillna('N/A')
    print('Cleaned field: vehicleManufacturerName ...')

    # Divide rates by 100 whenever the max is greater than divMax
    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    for r in const.rateFields():
        for s in secKeys:
            secIndx = np.where(dtPmts['securitizationKey'] == s)
            divInd = np.where(dtPmts[r].iloc[secIndx] > 1)
            if (len(divInd[0]) / len(secIndx[0]) > const.divInd()) and (dtPmts[r].iloc[secIndx].max() > const.divMax()):
                print('Plurality of values are greater than divMax for sec: %s, field: %r and dividing by 100 ...' %(s,r))
                dtPmts[r].iloc[secIndx] = dtPmts[r].iloc[secIndx] / 100
            elif (dtPmts[r].iloc[secIndx].max() > const.divMax()):
                divIndx = np.where(dtPmts[r].iloc[secIndx] > const.divMax())
                dtPmts[r].iloc[secIndx[0][divIndx[0]]] = np.nan
                print('Only some values are greater than divMax for sec: %s, field: %r and converting those to nan ...' %(s,r))
        divIndx = np.where(dtPmts[r] > 1)
        dtPmts[r].iloc[divIndx] = 0

    print('Fixed rate scaling ...')

    # Fix servicing fee amount
    servIndx = np.where(np.abs(dtPmts['servicingFeePercentage']) < const.minSens())
    dtPmts['servicingFeePercentage'].iloc[servIndx] = .01

    # Need to fix modificationTypeCode, subvented
    #dtPmts['modificationTypeCode'] = pd.to_numeric(dtPmts['modificationTypeCode'], errors='coerce')

    return dtPmts

def append_calc_fields(dtPmts):

    # Fix missing beginning balances

    begIndx = np.where(dtPmts['reportingPeriodBeginningLoanBalanceAmount'].isnull())
    dtPmts['reportingPeriodBeginningLoanBalanceAmount'].iloc[begIndx] = dtPmts[['reportingPeriodActualEndBalanceAmount','otherPrincipalAdjustmentAmount',
                                                                                'chargedoffPrincipalAmount','actualPrincipalCollectedAmount']].iloc[begIndx].sum(axis=1)

    # Fix charge-offs so payments and balances tie out

    adjIndx = np.logical_or.reduce([np.abs(dtPmts['repossessedProceedsAmount']) > const.minSens(),
                                     np.abs(dtPmts['chargedoffPrincipalAmount']) > const.minSens(),
                                     np.abs(dtPmts['recoveredAmount']) > const.minSens()])

    # If coAmt = openBal, and recovAmt = 0 and repoAmt > 0,
    # then recovAmt += repoAmt
    coIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['chargedoffPrincipalAmount'] - dtPmts['reportingPeriodBeginningLoanBalanceAmount']) < const.minSens(),
                                             np.abs(dtPmts['recoveredAmount']) < const.minSens(),
                                             np.abs(dtPmts['repossessedProceedsAmount']) > const.minSens(),
                                             adjIndx]))
    dtPmts['recoveredAmount'].iloc[coIndx] += dtPmts['repossessedProceedsAmount'].iloc[coIndx]

    # If repoAmt + coAmt + recovAmt = openBal,
    # then coAmt += repoAmt + recovAmt
    # then recovAmt += repoAmt
    # then endBal = 0
    coIndx = np.where(np.logical_and(np.abs(dtPmts[['repossessedProceedsAmount','chargedoffPrincipalAmount','recoveredAmount']].sum(axis=1)
                                            - dtPmts['reportingPeriodBeginningLoanBalanceAmount']) < const.minSens(),adjIndx))
    dtPmts['chargedoffPrincipalAmount'].iloc[coIndx] += dtPmts[['repossessedProceedsAmount','recoveredAmount']].iloc[coIndx].sum(axis=1)
    dtPmts['recoveredAmount'].iloc[coIndx] += dtPmts['repossessedProceedsAmount'].iloc[coIndx]
    dtPmts['reportingPeriodActualEndBalanceAmount'].iloc[coIndx] = 0

    # Ensure payments and balances tick and tie
    prinOffset = (dtPmts['reportingPeriodBeginningLoanBalanceAmount'] - dtPmts['reportingPeriodActualEndBalanceAmount']) - \
                 (dtPmts['actualPrincipalCollectedAmount'] + dtPmts['chargedoffPrincipalAmount'] + dtPmts['otherPrincipalAdjustmentAmount'])

    dtPmts['otherPrincipalAdjustmentAmount'] += prinOffset
    print('Fixed principalAdjustmentAmount ...')

    # Insert latest row indicator and add months & age from cutoff date, balance at cutoff, and shelf
    monthsFromCutoff = np.zeros(shape=(dtPmts.shape[0], 1))
    summaryDate = np.zeros(shape=(dtPmts.shape[0], 1))
    secKeys = np.sort(dtPmts['securitizationKey'].unique())

    for sec in secKeys:
        maxDate = np.max(dtPmts['reportingPeriodEndingDate'].iloc[np.where(dtPmts['securitizationKey'] == sec)])
        maxIndx = np.where(np.logical_and(dtPmts['securitizationKey'] == sec, dtPmts['reportingPeriodEndingDate'] == maxDate))
        summaryDate[maxIndx, 0] = 1

        secIndx = np.where(dtPmts['securitizationKey'] == sec)
        cutoffDate = dtPmts['reportingPeriodBeginningDate'].iloc[secIndx].min()
        monthsFromCutoff[secIndx, 0] = (12 * dtPmts['reportingPeriodBeginningDate'].iloc[secIndx].dt.year +
                                        dtPmts['reportingPeriodBeginningDate'].iloc[secIndx].dt.month).values - \
                                       12 * cutoffDate.year - cutoffDate.month

    dtPmts['summaryDate'] = summaryDate
    print('Inserted last date of data per securitization ...')

    # Add loan age
    dtPmts['age'] = 12 * dtPmts['reportingPeriodBeginningDate'].dt.year + dtPmts['reportingPeriodBeginningDate'].dt.month - \
                    12 * dtPmts['originationDate'].dt.year - dtPmts['originationDate'].dt.month
    print('Added field: age ...')

    # Add beginningBalance, age, months from CutoffDate
    dtPmts['monthsFromCutoffDate'] = monthsFromCutoff
    dtPmts['ageFromCutoffDate'] = dtPmts['age'] - dtPmts['monthsFromCutoffDate']
    dtMatch = dtPmts[['assetNumber','securitizationKey','reportingPeriodBeginningLoanBalanceAmount']].iloc[np.where(dtPmts['monthsFromCutoffDate'] == 0)]
    dtPmts = pd.merge(dtPmts,dtMatch,how='left',left_on=['assetNumber','securitizationKey'],right_on=['assetNumber','securitizationKey'],copy=False,suffixes=['','_y'])
    dtPmts = dtPmts.rename(columns = {'reportingPeriodBeginningLoanBalanceAmount_y':'beginningBalanceAtCutoffDate'})
    print('Added fields: beginningBalanceAtCutoffDate, ageFromCutoffDate, monthsFromCutoffDate ...')

    # Determine if score is consumer, commercial, or other
    consumerCreditScore = np.expand_dims(dtPmts['obligorCreditScore'].values,axis=1)
    commercialCreditScore = np.ones(shape=(dtPmts.shape[0],1)) * np.nan

    commIndx = np.where(dtPmts['obligorCreditScoreType'].str.contains('commercial', case=False))
    commercialCreditScore[commIndx,0] = dtPmts['obligorCreditScore'].iloc[commIndx].values
    otherIndx = np.where(np.logical_or.reduce([dtPmts['obligorCreditScore'] < 300, dtPmts['obligorCreditScore'] > 850,
                                               dtPmts['obligorCreditScoreType'].str.contains('Unknown/Invalid', case=False),
                                               dtPmts['obligorCreditScoreType'].str.contains('None', case=False)]))

    consumerCreditScore[otherIndx,0] = np.nan
    commercialCreditScore[otherIndx, 0] = np.nan
    consumerCreditScore[commIndx, 0] = np.nan

    dtPmts['consumerCreditScore'] = consumerCreditScore
    dtPmts['commercialCreditScore'] = commercialCreditScore
    print('Seperated consumer and commercial credit scores ...')

    # Add LTV, where V = 0, make it NaN
    dtPmts['loanToValueRatio'] = np.divide(dtPmts['originalLoanAmount'],dtPmts['vehicleValueAmount'])
    dtPmts['loanToValueRatio'].iloc[np.where(dtPmts['loanToValueRatio'] == np.inf)] = np.nan
    dtPmts['loanToValueRatio'].iloc[np.where(dtPmts['loanToValueRatio'] > 2)] = np.nan
    print('Added field: LTV ...')

    # Add vintage
    dtPmts['vintage'] = dtPmts['originationDate'].dt.year
    print('Added field: vintage ...')

    # Add monthsDelinquent, chargeOff and prepay indicator
    dtPmts['monthsDelinquent'] = np.ceil(dtPmts['currentDelinquencyStatus'] / 30)
    dtPmts['monthsDelinquent'].iloc[np.where(dtPmts['monthsDelinquent'] > 4)] = 4
    dtPmts['monthsDelinquent'].iloc[np.where(dtPmts['chargedoffPrincipalAmount'] > 0)] = 5
    dtPmts['monthsDelinquent'].iloc[np.where(np.logical_and(dtPmts['otherPrincipalAdjustmentAmount'] + dtPmts['actualPrincipalCollectedAmount'] >= dtPmts['reportingPeriodBeginningLoanBalanceAmount'],
                                                            dtPmts['reportingPeriodActualEndBalanceAmount'] <= .05))] = 6
    print('Added field: monthsDelinquent ...')

    # Add dueAmount and principalPrepaid
    dtNextPmtDue = dtPmts[['assetNumber','securitizationKey','nextReportingPeriodPaymentAmountDue']].copy(deep=True)
    dtNextPmtDue['nextReportingPeriod'] = (pd.DatetimeIndex(dtPmts['reportingPeriodEndingDate']) + pd.DateOffset(1)).strftime('%m-%d-%Y')
    dtPmts['nextReportingPeriod'] = pd.DatetimeIndex(dtPmts['reportingPeriodBeginningDate']).strftime('%m-%d-%Y')
    dtPmts = pd.merge(dtPmts,dtNextPmtDue,how='left',left_on=['assetNumber','securitizationKey','nextReportingPeriod'],
                      right_on=['assetNumber','securitizationKey','nextReportingPeriod'],copy=False,suffixes=['','_y'])

    dtPmts = dtPmts.drop(['nextReportingPeriod'],axis=1)
    dtPmts = dtPmts.rename(columns = {'nextReportingPeriodPaymentAmountDue_y':'dueAmount'})
    dtPmts['dueAmount'] = pd.to_numeric(dtPmts['dueAmount'], errors='coerce')
    nanIndx = np.where(dtPmts['dueAmount'].isnull())
    dtPmts['dueAmount'].iloc[nanIndx] = dtPmts[['scheduledInterestAmount','scheduledPrincipalAmount']].iloc[nanIndx].sum(axis=1) * \
                                        (np.floor(dtPmts['currentDelinquencyStatus'].iloc[nanIndx]/30) + 1)

    dtPmts['principalPrepaid'] = np.maximum(dtPmts[['actualPrincipalCollectedAmount','actualInterestCollectedAmount']].sum(axis=1) - dtPmts['dueAmount'],0)
    pmtIndx = np.where(np.abs(dtPmts['chargedoffPrincipalAmount']) > const.minSens())
    # No prepayment when there is even a partial charge off
    dtPmts['principalPrepaid'].iloc[pmtIndx] = 0
    print('Added fields: dueAmount and principalPrepaid ...')

    # Add prime signifiers
    prime = np.zeros(shape=(dtPmts.shape[0],1))
    # Superprime is > 740
    prime[np.where(np.logical_and(dtPmts['consumerCreditScore']>740,
                                  np.isnan(dtPmts['consumerCreditScore'])==False)),0] = 4
    # Prime is 680 - 740
    prime[np.where(np.logical_and.reduce([dtPmts['consumerCreditScore']>680,
                                          dtPmts['consumerCreditScore']<=740,
                                          np.isnan(dtPmts['consumerCreditScore'])==False])),0] = 3
    # Near prime is 640 - 680
    prime[np.where(np.logical_and.reduce([dtPmts['consumerCreditScore']>640,
                                          dtPmts['consumerCreditScore']<=680,
                                          np.isnan(dtPmts['consumerCreditScore'])==False])),0] = 2
    # Sub prime is
    prime[np.where(np.logical_and(dtPmts['consumerCreditScore']<= 640,
                                  np.isnan(dtPmts['consumerCreditScore'])==False)),0] = 1
    # Other is 0
    dtPmts['primeIndicator'] = prime
    print('Added field: primeIndicator ...')

    # Add net losses
    dtPmts['netLosses'] = np.maximum(dtPmts['chargedoffPrincipalAmount'] - dtPmts['recoveredAmount'],0).fillna(0)
    print('Added field: netLosses ...')

    #Add region
    dtRegion = read_dict('states.csv')
    dtPmts = pd.merge(dtPmts, dtRegion, how='left', left_on='obligorGeographicLocation', right_on='state', copy=False)
    dtPmts = dtPmts.drop(['state'], axis=1)
    print('Added field: region ...')

    return dtPmts

def data_vetting(dtPmts):
#Desc: Finds initial set of errors with raw (minimally processed) data.

    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    numFields = const.decimalFields() + const.integerFields() + const.rateFields() + const.dateFields()
    strFields = const.stringFields()
    dtErrors = pd.DataFrame(data=np.zeros(shape=(len(const.rawCols()),len(secKeys))),index=const.rawCols(),columns=secKeys)
    dtCF = pd.DataFrame()
    dtDescNum = pd.DataFrame()
    dtDescStr = pd.DataFrame()

    for s in secKeys:
        print('Identifying errors in securitization: %s ... ' %s)
        dtP = dtPmts.iloc[np.where(dtPmts['securitizationKey'] == s)]

        #dtErrors tracks errors in the raw data at the deal level
        dtErrors[s].ix['Count'] = len(dtP['assetNumber'].unique())
        dtErrors[s].ix['OpenBal'] = dtP['reportingPeriodBeginningLoanBalanceAmount'].iloc[
            np.where(dtP['reportingPeriodBeginningDate'] == dtP['reportingPeriodBeginningDate'].unique().min())].sum()
        dtErrors[s].ix['StartMonth'] = dtP['reportingPeriodBeginningDate'].min()
        dtErrors[s].ix['EndMonth'] = dtP['reportingPeriodBeginningDate'].max()
        dtErrors[s].ix['MissingMonths'] = len(np.where(np.in1d(np.array(pd.date_range(start=dtP['reportingPeriodEndingDate'].min(),end=dtPmts['reportingPeriodEndingDate'].max(),freq='M')),
                                                    dtP['reportingPeriodEndingDate'].unique()) == False)[0])

        dtErrors[s].ix['Walk'] = len(np.where(np.logical_and.reduce([dtP['reportingPeriodBeginningLoanBalanceAmount'].iloc[1:] != dtP['reportingPeriodActualEndBalanceAmount'].iloc[:-1],
                                                  dtP['assetNumber'].iloc[1:] == dtP['assetNumber'].iloc[:-1],
                                                  dtP['securitizationKey'].iloc[1:] == dtP['securitizationKey'].iloc[:-1]]))[0])
        dtErrors[s].ix['IncrBal'] = len(np.where(np.logical_and.reduce([dtP['reportingPeriodBeginningLoanBalanceAmount'].iloc[:-1] < dtP['reportingPeriodBeginningLoanBalanceAmount'].iloc[1:],
                                                  dtP['assetNumber'].iloc[1:] == dtP['assetNumber'].iloc[:-1],
                                                  dtP['securitizationKey'].iloc[1:] == dtP['securitizationKey'].iloc[:-1]]))[0])
        dtErrors[s].ix['Pmts'] = len(np.where(np.abs((dtP['reportingPeriodBeginningLoanBalanceAmount'] - dtP['reportingPeriodActualEndBalanceAmount']) - \
                         (dtP['actualPrincipalCollectedAmount'] + dtP['chargedoffPrincipalAmount'] + dtP['otherPrincipalAdjustmentAmount'])) > const.minSens())[0])

        # To calculate missing and extra records we compare MoM assetNumbers
        secMonths = np.sort(dtP['reportingPeriodBeginningDate'].unique())
        if (len(secMonths) > 1):
            for i in range(0,len(secMonths)-1):
                lastIndx = np.where(np.logical_and(dtP['reportingPeriodBeginningDate'] == secMonths[i],np.abs(dtP['reportingPeriodActualEndBalanceAmount']) > const.minSens()))
                nextlastIndx = np.where(dtP['reportingPeriodBeginningDate'] == secMonths[i+1])

                lastLoans = dtP['assetNumber'].iloc[lastIndx].unique()
                nextlastLoans = dtP['assetNumber'].iloc[nextlastIndx].unique()

                dtErrors[s].ix['Missing'] += len(np.where(np.in1d(lastLoans,nextlastLoans) == False)[0])
                dtErrors[s].ix['Extra'] += len(np.where(np.in1d(nextlastLoans, lastLoans) == False)[0])

        for c in np.where(dtP['chargedoffPrincipalAmount'] > const.minSens())[0]:
            dtErrors[s].ix['COExtra'] += len(np.where(np.logical_and(dtP['assetNumber'] == dtP['assetNumber'].iloc[c],
                                    dtP['reportingPeriodBeginningDate'] > dtP['reportingPeriodBeginningDate'].iloc[c]))[0])

        dtErrors[s].ix['Dupes'] = len(dtP.set_index(['assetNumber', 'reportingPeriodBeginningDate', 'securitizationKey']).index.get_duplicates())
        dtErrors[s].ix['NegOpenBal'] = len(np.where(dtP['reportingPeriodBeginningLoanBalanceAmount'] < 0)[0])
        dtErrors[s].ix['NegCloseBal'] = len(np.where(dtP['reportingPeriodActualEndBalanceAmount'] < 0)[0])
        dtErrors[s].ix['RateNeg'] = len(np.where(dtP[const.rateFields()] < 0)[0])
        dtErrors[s].ix['RatePos'] = len(np.where(dtP[const.rateFields()] > 1)[0])
        dtErrors[s].ix['Integer'] = len(np.where(dtP[const.integerFields()].mod(1,axis=0,fill_value=0) != 0)[0])
        dtErrors[s].ix['NegCO'] = len(np.where(dtP['chargedoffPrincipalAmount'] < 0)[0])
        dtErrors[s].ix['PartialCO'] = len(np.where(np.logical_and(dtP['chargedoffPrincipalAmount'] < dtP['reportingPeriodBeginningLoanBalanceAmount'],
                                                                  np.abs(dtP['chargedoffPrincipalAmount']) > const.minSens()))[0])
        dtErrors[s].ix['GreaterCO'] = len(np.where(np.logical_and(dtP['chargedoffPrincipalAmount'] > dtP['reportingPeriodBeginningLoanBalanceAmount'],
                                                                  np.abs(dtP['chargedoffPrincipalAmount']) > const.minSens()))[0])
        dtErrors[s].ix['NegRepo'] = len(np.where(dtP['repossessedProceedsAmount'] < 0)[0])
        dtErrors[s].ix['NegRecov'] = len(np.where(dtP['recoveredAmount'] < 0)[0])

        dtCF = pd.concat([dtCF, cashflow_vetting(dtP).sum(axis=0).transpose()], axis=1)

    dtCF.columns = secKeys
    dtErrors = pd.concat([dtErrors,dtCF],axis=0)

# dtDescNum and dtDescStr tracks errors for numerical and string fields at the field level
    dtTemp = pd.concat([dtP[numFields].describe(include=[np.number]).transpose(),dtP[numFields].isnull().sum(axis=0),
                        (dtP[numFields] == 0).sum(axis=0),(dtP[const.rateFields()] > 1).sum(axis=0),
                        (dtP[const.rateFields()] < 0).sum(axis=0),
                        (dtP[const.integerFields()].mod(1,axis=0,fill_value=0) != 0).sum(axis=0)],axis=1)
    dtTemp.columns = s + ' ' + np.append(dtP[numFields].describe(include=[np.number]).index.values,
                                         (['nans','zeros','rate>1','rate<0','non-int']))
    dtDescNum = pd.concat([dtDescNum,dtTemp],axis=1)

    dtTemp = pd.concat([dtPmts[strFields].describe(exclude=[np.number]).transpose(),
                        dtP[strFields].isnull().sum(axis=0)], axis=1)
    dtTemp.columns = s + ' ' + np.append(dtPmts[strFields].describe(exclude=[np.number]).index,'nans')
    dtDescStr = pd.concat([dtDescStr,dtTemp],axis=1)
    print('Processed raw errors for securitization: %s ... '%s)

    return dtErrors,dtDescNum,dtDescStr

def cashflow_vetting(dtP):

    return dtP.groupby(['reportingPeriodBeginningDate'])['reportingPeriodBeginningLoanBalanceAmount','actualPrincipalCollectedAmount',
                               'chargedoffPrincipalAmount','otherPrincipalAdjustmentAmount',
                               'reportingPeriodActualEndBalanceAmount','actualInterestCollectedAmount',#'principalPrepaid',
                               'recoveredAmount','repossessedProceedsAmount'].sum()

def main(argv = sys.argv):

dtClean = clean_ald_files(dtRaw)
dtPmts = append_calc_fields(dtClean)
pmtIndx = np.where(np.abs(dtPmts['chargedoffPrincipalAmount']) > .01)
lIndx = dtPmts['assetNumber'].iloc[pmtIndx].unique()
pmtIndx = np.where(np.in1d(dtPmts['assetNumber'],lIndx))
rawIndx = np.where(np.in1d(dtRaw['assetNumber'],lIndx))

dtRaw.iloc[rawIndx].to_csv('raw.csv')
dtPmts.iloc[pmtIndx].to_csv('clean.csv')


kk = pd.pivot_table(data=dtPmts,values='securitizationKey',index='obligorIncomeVerificationLevelCode',columns='obligorEmploymentVerificationCode',aggfunc=np.count_nonzero)


dtPmts.groupby(['securitizationKey','reportingPeriodBeginningDate'])[['principalPrepaid','otherPrincipalAdjustmentAmount',
                                                                      'chargedoffPrincipalAmount','recoveredAmount']].sum().to_csv('out.csv')

dtErrors,dtDescNum,dtDescStr = data_vetting(dtPmts)
#dtDescNum.to_csv('descnum.csv')
#dtDescStr.to_csv('descstr.csv')
dtErrors.to_csv('Error Log/cleanErrors20170508.csv')


coIndx = np.where(np.abs(dtPmts['chargedoffPrincipalAmount']) > const.minSens())
coIndx = np.where(np.in1d(dtPmts['assetNumber'],dtPmts['assetNumber'].iloc[coIndx]))
dtPmts[const.debugFieldsClean()].iloc[coIndx].to_csv('cleanCO.csv')

ii = np.where(np.in1d(dtRaw['assetNumber'],dtPmts['assetNumber'].iloc[coIndx]))
dtRaw[const.debugFieldsRaw()].iloc[ii].to_csv('raw.csv')

ii = np.where(dtPmts['assetNumber'] == '368159')

if __name__ == "__main__":
    sys.exit(main())