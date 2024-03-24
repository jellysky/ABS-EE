import pandas as pd
import numpy as np

class const():
    @staticmethod
    def booleanFields():
        return ['assetAddedIndicator','assetSubjectDemandIndicator','coLesseePresentIndicator','reportingPeriodModificationIndicator',
                'underwritingIndicator']
    @staticmethod
    def dateFields():
        return ['originalFirstPaymentDate','originationDate','paidThroughDate','reportingPeriodBeginDate','reportingPeriodEndDate',
                'scheduledTerminationDate','zeroBalanceEffectiveDate']
    @staticmethod
    def debugFieldsRaw():
        return ['assetNumber', 'securitizationKey', 'reportingPeriodBeginDate','reportingPeriodEndDate','reportingPeriodEndingActualBalanceAmount',
                'totalActualAmountPaid','reportingPeriodScheduledPaymentAmount','contractResidualValue',
                'acquisitionCost', 'vehicleValueAmount', 'reportingPeriodSecuritizationValueAmount',
                'reportingPeriodEndActualSecuritizationAmount', 'chargedOffAmount', 'liquidationProceedsAmount',
                'repurchaseAmount', 'baseResidualValue','securitizationDiscountRate', 'currentDelinquencyStatus',
                'remainingTermNumber', 'originalLeaseTermNumber']
    @staticmethod
    def debugFieldsClean():
        return ['assetNumber','securitizationKey','reportingPeriodBeginDate','reportingPeriodEndingActualBalanceAmount',
                'totalActualAmountPaid','actualOtherCollectedAmount','reportingPeriodScheduledPaymentAmount','contractResidualValue',
                'acquisitionCost','vehicleValueAmount','reportingPeriodSecuritizationValueAmount','reportingPeriodEndActualSecuritizationAmount',
                'actualPrincipalCollectedAmount','actualInterestCollectedAmount','otherPrincipalAdjustmentAmount','principalPrepaid','chargedOffAmount','recoveredAmount','liquidationProceedsAmount',
                'scheduledSecuritizationBeginValueAmount','scheduledSecuritizationEndValueAmount','scheduledSecuritizationValueAmortization','scheduledSecuritizationValueInterest',
                'securitizationDiscountRate','baseResidualValue','originalInterestRatePercentage','currentDelinquencyStatus','monthsDelinquent',
                'remainingTermNumber','originalLeaseTermNumber','terminationIndicator','nextReportingPeriodPaymentAmountDue',
                'lesseeGeographicLocation','saleGainOrLoss','zeroBalanceCode']
    @staticmethod
    def decimalFields():
        return ['acquisitionCost','actualOtherCollectedAmount','baseResidualValue','chargedOffAmount','contractResidualValue',
                'excessFeeAmount','liquidationProceedsAmount','nextReportingPeriodPaymentAmountDue','otherAssessedUncollectedServicerFeeAmount',
                'otherLeaseLevelServicingFeesRetainedAmount','reportingPeriodEndActualSecuritizationAmount','reportingPeriodEndingActualBalanceAmount',
                'reportingPeriodScheduledPaymentAmount','reportingPeriodSecuritizationValueAmount','repurchaseAmount','servicerAdvancedAmount',
                'servicingFlatFeeAmount','totalActualAmountPaid','vehicleValueAmount']
    @staticmethod
    def integerFields():
        return ['baseResidualSourceCode','currentDelinquencyStatus','gracePeriod','leaseExtended','lesseeCreditScore',
                'lesseeEmploymentVerificationCode','lesseeIncomeVerificationLevelCode','modificationTypeCode',
                'originalLeaseTermNumber','paymentTypeCode','remainingTermNumber','servicingAdvanceMethodCode','terminationIndicator',
                'vehicleModelYear','vehicleNewUsedCode','vehicleTypeCode','vehicleValueSourceCode','zeroBalanceCode']
    @staticmethod
    def listFields():
        return ['subvented']
    @staticmethod
    def rateFields():
        return ['paymentToIncomePercentage','securitizationDiscountRate','servicingFeePercentage']
    @staticmethod
    def stringFields():
        return ['assetNumber','assetTypeNumber','lesseeCreditScoreType','lesseeGeographicLocation','originatorName',
                'primaryLeaseServicerName','securitizationKey','shelf','vehicleManufacturerName','vehicleModelName']
    @staticmethod
    def rawCols():
        return ['Count','OpenBal','StartMonth','EndMonth','MissingMonths','Walk','IncrBal','Pmts','Missing','Extra',
                'COExtra','Dupes','NegOpenBal','NegCloseBal','RateNeg','RatePos','Integer','NegCO','PartialCO','GreaterCO','NegRepo','NegRecov']
    @staticmethod
    def minSens():
        return .01
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
# Inputs: The raw ALD payment dataframe
# Outputs: The cleaned ALD payment dataframe
# Desc: Cleans raw ald file data and puts it into a format that is usable for reporting

    # dtPmts['shelf'] = dtPmts['securitizationKey']
    # secKeys = np.sort(dtPmts['securitizationKey'].unique())
    # for s in secKeys:
    #     secIndx = np.where(dtPmts['securitizationKey'] == s)
    #     dtPmts['shelf'].iloc[secIndx] = s[:s.find('2017') - 1]
    # print('Added shelf ... ')

    # Drop Daimler Trust LLC
    dropIndx = np.where(dtPmts['securitizationKey'] != 'DAIMLER TRUST LEASING LLC')
    dtPmts = dtPmts.iloc[dropIndx]

    # Convert date to datetime objects
    dtPmts['reportingPeriodBeginDate'] = pd.to_datetime(dtPmts['reportingPeriodBeginDate'], format='%m-%d-%Y')
    dtPmts['reportingPeriodEndDate'] = pd.to_datetime(dtPmts['reportingPeriodEndDate'], format='%m-%d-%Y')
    dtPmts['paidThroughDate'] = pd.to_datetime(dtPmts['paidThroughDate'], format='%m-%d-%Y')
    dtPmts['originationDate'] = pd.to_datetime(dtPmts['originationDate'], format='%m/%Y')
    dtPmts['originalFirstPaymentDate'] = pd.to_datetime(dtPmts['originalFirstPaymentDate'], format='%m/%Y')
    dtPmts['zeroBalanceEffectiveDate'] = pd.to_datetime(dtPmts['zeroBalanceEffectiveDate'], format='%m/%Y')
    dtPmts['scheduledTerminationDate'] = pd.to_datetime(dtPmts['scheduledTerminationDate'], format='%m/%Y')

    print('Cleaned dates ...')

    # Replace strings with numbers (No score = NaN)
    replaceStr = const.decimalFields() + const.integerFields() + const.rateFields()

    for r in replaceStr:
        dtPmts[r] = pd.to_numeric(dtPmts[r], errors='coerce')
        print('Cleaned field: %s ...' % r)

    # Replace nans on certain fields to 0
    nanStr = np.sort(['chargedOffAmount','contractResidualValue','excessFeeAmount','liquidationProceedsAmount',
              'otherAssessedUncollectedServicerFeeAmount','otherLeaseLevelServicingFeesRetainedAmount',
              'totalActualAmountPaid','lesseeIncomeVerificationLevelCode','lesseeEmploymentVerificationCode',
              'lesseeCreditScore','repurchaseAmount','actualOtherCollectedAmount','currentDelinquencyStatus',
              'terminationIndicator','zeroBalanceCode','paymentToIncomePercentage'])
    for n in nanStr:
        dtPmts[n] = dtPmts[n].fillna(0)
    print('Cleaned nans ...')

    # Fix double months in same row
    pmtIndx = np.where(12 * dtPmts['reportingPeriodEndDate'].dt.year + dtPmts['reportingPeriodEndDate'].dt.month -
                       12 * dtPmts['reportingPeriodBeginDate'].dt.year - dtPmts['reportingPeriodBeginDate'].dt.month == 1)
    dtPmts['reportingPeriodBeginDate'].iloc[pmtIndx] = (pd.DatetimeIndex(dtPmts['reportingPeriodBeginDate'].iloc[pmtIndx]) + pd.DateOffset(months=1))

    dtLast = dtPmts[['assetNumber','securitizationKey','reportingPeriodEndActualSecuritizationAmount']]
    dtLast['reportingPeriodBeginDate'] = (pd.DatetimeIndex(dtPmts['reportingPeriodBeginDate']) + pd.DateOffset(months=1))
    dtPmts = pd.merge(dtPmts,dtLast,how='left',left_on=['assetNumber','securitizationKey','reportingPeriodBeginDate'],
                      right_on=['assetNumber','securitizationKey','reportingPeriodBeginDate'],copy=False,suffixes=['','_y'])
    dtPmts['reportingPeriodSecuritizationValueAmount'].iloc[pmtIndx] = dtPmts['reportingPeriodEndActualSecuritizationAmount_y'].iloc[pmtIndx]
    dtPmts = dtPmts.drop(['reportingPeriodEndActualSecuritizationAmount_y'],axis=1)
    print('Cleaned double month rows ... ')

    # Drop duplicates - Carmax u vs b
    dtPmts = dtPmts.drop_duplicates(subset=['assetNumber','reportingPeriodBeginDate','securitizationKey'],keep='last')
    dtPmts = dtPmts.sort_values(by=['assetNumber','reportingPeriodBeginDate','securitizationKey'],axis=0)
    dtPmts = dtPmts.reset_index(drop=True)
    print('Dropped duplicate records and reindexed ...')

    # Fix manufacturer names
    dtManus = read_dict('manus.csv')
    dtPmts = pd.merge(dtPmts, dtManus, how='left', left_on='vehicleManufacturerName', right_on='old', copy=False)
    dtPmts = dtPmts.drop(['old', 'vehicleManufacturerName'], axis=1)
    dtPmts = dtPmts.rename(columns={'new': 'vehicleManufacturerName'})
    dtPmts['vehicleManufacturerName'] = dtPmts['vehicleManufacturerName'].fillna('N/A')
    print('Cleaned field: vehicleManufacturerName ...')

    # Fix model names
    dtModel = read_dict('model.csv')
    dtPmts = pd.merge(dtPmts, dtModel, how='left', left_on='vehicleModelName', right_on='old', copy=False)
    dtPmts = dtPmts.drop(['old', 'vehicleModelName'], axis=1)
    dtPmts = dtPmts.rename(columns={'new': 'vehicleModelName'})
    dtPmts['vehicleModelName'] = dtPmts['vehicleModelName'].fillna('N/A')
    print('Cleaned field: vehicleModelName ...')

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

    # Calculate schOpenSecBal and schCloseSecBal for leases
    dtPmts['scheduledSecuritizationBeginValueAmount'] = np.pv(dtPmts['securitizationDiscountRate']/12,dtPmts['remainingTermNumber']+1,
                                                               -dtPmts['reportingPeriodScheduledPaymentAmount'],-dtPmts['baseResidualValue'],1)
    dtPmts['scheduledSecuritizationEndValueAmount'] = np.pv(dtPmts['securitizationDiscountRate']/12,dtPmts['remainingTermNumber'],
                                                               -dtPmts['reportingPeriodScheduledPaymentAmount'],-dtPmts['baseResidualValue'],1)

    # ChargeOff and openSecBal adjustments
    # Condition 1: Where openSecBal = 0 & closeSecBal != 0 & chargeOff < 0 & remTerm is > 0,
    # then chargeOff = openSecBal - closeSecBal
    coIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['reportingPeriodSecuritizationValueAmount']) < const.minSens(),
                                              np.abs(dtPmts['reportingPeriodEndActualSecuritizationAmount']) > const.minSens(),
                                              dtPmts['chargedOffAmount'] < 0,
                                              dtPmts['remainingTermNumber'] > 0]))

    dtPmts['chargedOffAmount'].iloc[coIndx] = dtPmts['reportingPeriodSecuritizationValueAmount'].iloc[coIndx] - dtPmts['reportingPeriodEndActualSecuritizationAmount'].iloc[coIndx]

    # Condition 2: Where openSecBal = 0 & closeSecBal != 0 & chargeOff >= 0 & remTerm is > 0,
    # then openSecBal = closeSecBal + chargeOff
    begIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['reportingPeriodSecuritizationValueAmount']) < const.minSens(),
                                              np.abs(dtPmts['reportingPeriodEndActualSecuritizationAmount']) > const.minSens(),
                                              dtPmts['chargedOffAmount'] > 0,
                                              dtPmts['remainingTermNumber'] > 0]))
    dtPmts['reportingPeriodSecuritizationValueAmount'].iloc[begIndx] = dtPmts[['reportingPeriodEndActualSecuritizationAmount','chargedOffAmount']].iloc[begIndx].sum(axis=1)

    # Condition 3: Where chargedOff > 0 & chargeOff + liquidation = openSecBal,
    # then chargeOff = openSecBal
    coIndx = np.where(np.logical_and(dtPmts['chargedOffAmount'] > 0,
                                     np.abs(dtPmts['reportingPeriodSecuritizationValueAmount'] - dtPmts['chargedOffAmount'] - dtPmts['liquidationProceedsAmount']) < const.minSens()))
    dtPmts['chargedOffAmount'].iloc[coIndx] = dtPmts['reportingPeriodSecuritizationValueAmount'].iloc[coIndx]

    print('Fixed: scheduledSecuritizationAmounts, chargedOffAmounts, and added: reportedSecuritizationAmounts ... ')

    # Recovery or liquidation adjustments
    # Condition 1: Where chargedOff > 0 & liquidation > 0,
    # then recov = liquidation
    dtPmts['recoveredAmount'] = 0
    recovIndx = np.where(np.logical_and(dtPmts['chargedOffAmount'] > 0, dtPmts['liquidationProceedsAmount'] > 0))
    dtPmts['recoveredAmount'].iloc[recovIndx] = dtPmts['liquidationProceedsAmount'].iloc[recovIndx]

    # Condition 2: Where liquidation > 0 & zeroBalEffDate < currentDate & openSecBal + closeSecBal = 0, zeroBalCode = 3
    # then set recoveredAmount += liquidation
    recovIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['liquidationProceedsAmount']) > const.minSens(),
                                                dtPmts['zeroBalanceEffectiveDate'] < dtPmts['reportingPeriodBeginDate'],
                                                np.abs(dtPmts[['reportingPeriodEndingActualBalanceAmount','reportingPeriodEndActualSecuritizationAmount']].sum(axis=1)) < const.minSens(),
                                                dtPmts['zeroBalanceCode'] == 3]))
    dtPmts['recoveredAmount'].iloc[recovIndx] += dtPmts['liquidationProceedsAmount'].iloc[recovIndx]

    # Condition 3: Where termInd = [1,2] or liquidation > 0 & zeroBalEffDate < currentDate & openSecBal + closeSecBal = 0, zeroBalCode = 1
    # then totalActualAmountPaid += liquidation
    pmtIndx = np.in1d(dtPmts['terminationIndicator'],[1,2])
    pmtIndx = np.where(np.logical_or(np.logical_and.reduce([np.abs(dtPmts['liquidationProceedsAmount']) > const.minSens(),
                                dtPmts['zeroBalanceEffectiveDate'] < dtPmts['reportingPeriodBeginDate'],
                                np.abs(dtPmts[['reportingPeriodEndingActualBalanceAmount','reportingPeriodEndActualSecuritizationAmount']].sum(axis=1)) < const.minSens(),
                                dtPmts['zeroBalanceCode'] == 1]),pmtIndx))
    dtPmts['totalActualAmountPaid'].iloc[pmtIndx] += dtPmts['liquidationProceedsAmount'].iloc[pmtIndx]

    # Condition 4: Where there are duplicate chargeOffs, recovs, and liqProceeds,
    # then set chargeOff = recovs = liquidation = 0
    zeroIndx = np.where(np.logical_and.reduce([dtPmts['assetNumber'].iloc[:-1] == dtPmts['assetNumber'].iloc[1:],
                                               dtPmts['liquidationProceedsAmount'].iloc[:-1] == dtPmts['liquidationProceedsAmount'].iloc[1:],
                                               np.abs(dtPmts['liquidationProceedsAmount'].iloc[:-1]) > const.minSens()]))
    dtPmts['recoveredAmount'].iloc[zeroIndx] = 0
    dtPmts['chargedOffAmount'].iloc[zeroIndx] = 0
    dtPmts['liquidationProceedsAmount'].iloc[zeroIndx] = 0
    dtPmts['zeroBalanceCode'].iloc[zeroIndx] = 0
    dtPmts['terminationIndicator'].iloc[zeroIndx] = 0

    print('Fixed: Recoveries and totalAmtPaid ... ')

    # Add monthsDelinquent, chargeOff and prepay indicator
    dtPmts['currentDelinquencyStatus'] = dtPmts['currentDelinquencyStatus'].fillna(0)
    dtPmts['monthsDelinquent'] = np.floor(dtPmts['currentDelinquencyStatus'] / 30)
    dtPmts['monthsDelinquent'].iloc[np.where(dtPmts['monthsDelinquent'] > 4)] = 4
    dtPmts['monthsDelinquent'].iloc[np.where(dtPmts['chargedOffAmount'] > 0)] = 5
    prepayIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['reportingPeriodEndingActualBalanceAmount']) < const.minSens(),
                                        dtPmts['remainingTermNumber'] < dtPmts['originalLeaseTermNumber'],
                                        dtPmts['totalActualAmountPaid'] > dtPmts['reportingPeriodScheduledPaymentAmount'],
                                        dtPmts['currentDelinquencyStatus'] == 0]))
    dtPmts['monthsDelinquent'].iloc[prepayIndx] = 6
    print('Added field: monthsDelinquent ...')

    # Calculate actualPrincipalCollected, actualInterestCollected, otherPrincipalAdj, and schPrinDue, schIntDue for securitization balances
    # Calculate principal due
    delqIndx = np.where(dtPmts['monthsDelinquent'] < 5)
    dtPmts['scheduledSecuritizationValueAmortization'] = 0
    dtPmts['scheduledSecuritizationValueAmortization'].iloc[delqIndx] = np.multiply(dtPmts['scheduledSecuritizationBeginValueAmount'].iloc[delqIndx]-dtPmts['scheduledSecuritizationEndValueAmount'].iloc[delqIndx],
                                                                     (np.floor(dtPmts['currentDelinquencyStatus'].iloc[delqIndx]/30) + 1))
    # Calculate interest due
    dtPmts['scheduledSecuritizationValueInterest'] = dtPmts['scheduledSecuritizationBeginValueAmount'] * dtPmts['securitizationDiscountRate']/12

    # Calculated principal paid
    dtPmts['actualPrincipalCollectedAmount'] =  dtPmts['reportingPeriodSecuritizationValueAmount'] - dtPmts['reportingPeriodEndActualSecuritizationAmount'] - dtPmts['chargedOffAmount']

    # Where coAmt > 0 & openSecBal > 0 & closeSecBal = 0 & remTerm > 1,
    # then calculate principalPrepaid
    pmtIndx = np.where(np.logical_and.reduce([np.abs(dtPmts['chargedOffAmount']) < const.minSens(),
                                              np.abs(dtPmts['reportingPeriodSecuritizationValueAmount']) > const.minSens(),
                                              np.abs(dtPmts['reportingPeriodEndActualSecuritizationAmount']) < const.minSens(),
                                              dtPmts['remainingTermNumber'] > 1]))
    dtPmts['principalPrepaid'] = 0
    dtPmts['principalPrepaid'].iloc[pmtIndx] = np.maximum(dtPmts['actualPrincipalCollectedAmount'].iloc[pmtIndx] - dtPmts['scheduledSecuritizationValueAmortization'].iloc[pmtIndx], 0)

    # Calculate interest paid as minimum 0 or maximum secDiscRate * openSecBal
    dtPmts['actualInterestCollectedAmount'] = 0
    dtPmts['actualInterestCollectedAmount'] = (dtPmts['totalActualAmountPaid'] - dtPmts['actualPrincipalCollectedAmount'])
    dtPmts['actualInterestCollectedAmount'].iloc[np.where(dtPmts['actualInterestCollectedAmount'] < 0)] = 0
    nonPrinIndx = np.where(dtPmts['actualInterestCollectedAmount'] > dtPmts['reportingPeriodSecuritizationValueAmount'] * dtPmts['securitizationDiscountRate']/12)
    dtPmts['actualInterestCollectedAmount'].iloc[nonPrinIndx] = (dtPmts['reportingPeriodSecuritizationValueAmount'] * dtPmts['securitizationDiscountRate']/12).iloc[nonPrinIndx]

    # Calculate other principal adjustment
    dtPmts['otherPrincipalAdjustmentAmount'] = dtPmts['totalActualAmountPaid'] - dtPmts['actualPrincipalCollectedAmount'] - dtPmts['actualInterestCollectedAmount']

    # Add gain or loss on sale
    dtPmts['saleGainOrLoss'] = 0
    gainIndx = np.where(np.logical_or(np.in1d(dtPmts['terminationIndicator'],[1, 2]),np.in1d(dtPmts['zeroBalanceCode'],[1,2])))
    dtPmts['saleGainOrLoss'].iloc[gainIndx] = (dtPmts['liquidationProceedsAmount'] - dtPmts['contractResidualValue']).iloc[gainIndx]

    print('Added fields: Interest, scheduled prin, and unscheduled prin paid and interest, scheduled prin due, gain or loss on sale ... ')


    # Determine if score is consumer, commercial, or other
    consumerCreditScore = np.expand_dims(dtPmts['lesseeCreditScore'].values,axis=1)
    commercialCreditScore = np.ones(shape=(dtPmts.shape[0],1)) * np.nan

    commIndx = np.where(dtPmts['lesseeCreditScoreType'].str.contains('commercial', case=False))
    commercialCreditScore[commIndx,0] = dtPmts['lesseeCreditScore'].iloc[commIndx].values
    otherIndx = np.where(np.logical_or.reduce([dtPmts['lesseeCreditScore'] < 300, dtPmts['lesseeCreditScore'] > 850,
                                               dtPmts['lesseeCreditScoreType'].str.contains('Unknown/Invalid', case=False),
                                               dtPmts['lesseeCreditScoreType'].str.contains('None', case=False)]))

    # Insert latest row indicator and add months & age from cutoff date, balance at cutoff
    monthsFromCutoff = np.zeros(shape=(dtPmts.shape[0], 1))
    summaryDate = np.zeros(shape=(dtPmts.shape[0], 1))
    secKeys = np.sort(dtPmts['securitizationKey'].unique())

    for sec in secKeys:
        maxDate = np.max(dtPmts['reportingPeriodEndDate'].iloc[np.where(dtPmts['securitizationKey'] == sec)])
        maxIndx = np.where(np.logical_and(dtPmts['securitizationKey'] == sec, dtPmts['reportingPeriodEndDate'] == maxDate))
        summaryDate[maxIndx, 0] = 1

        secIndx = np.where(dtPmts['securitizationKey'] == sec)
        cutoffDate = dtPmts['reportingPeriodBeginDate'].iloc[secIndx].min()
        monthsFromCutoff[secIndx, 0] = (12 * dtPmts['reportingPeriodBeginDate'].iloc[secIndx].dt.year +
                                        dtPmts['reportingPeriodBeginDate'].iloc[secIndx].dt.month).values - \
                                       12 * cutoffDate.year - cutoffDate.month

    dtPmts['age'] = dtPmts['originalLeaseTermNumber'] - dtPmts['remainingTermNumber']
    dtPmts['summaryDate'] = summaryDate
    dtPmts['monthsFromCutoffDate'] = monthsFromCutoff
    dtPmts['ageFromCutoffDate'] = dtPmts['age'] - dtPmts['monthsFromCutoffDate']
    dtMatch = dtPmts[['assetNumber','securitizationKey','reportingPeriodSecuritizationValueAmount']].iloc[np.where(dtPmts['monthsFromCutoffDate'] == 0)]
    dtPmts = pd.merge(dtPmts, dtMatch, how='left', left_on=['assetNumber', 'securitizationKey'],right_on=['assetNumber', 'securitizationKey'], copy=False,suffixes=['','_y'])
    dtPmts = dtPmts.rename(columns={'reportingPeriodSecuritizationValueAmount_y': 'beginningBalanceAtCutoffDate'})

    print('Added fields: age, summaryDate, monthsFromCutoffDate, ageFromCutoffDate, beginningBalanceAtCutoffDate ...')

    consumerCreditScore[otherIndx,0] = np.nan
    commercialCreditScore[otherIndx, 0] = np.nan
    consumerCreditScore[commIndx, 0] = np.nan

    dtPmts['consumerCreditScore'] = consumerCreditScore
    dtPmts['commercialCreditScore'] = commercialCreditScore
    print('Seperated consumer and commercial credit scores ...')

    # Add LTV, where V = 0, make it NaN
    dtPmts['loanToValueRatio'] = np.divide(dtPmts['acquisitionCost'],dtPmts['vehicleValueAmount'])
    dtPmts['loanToValueRatio'].iloc[np.where(dtPmts['loanToValueRatio'] == np.inf)] = np.nan
    print('Added field: LTV ...')

    # Add original loan amount
    dtPmts['originalLoanAmount'] = dtPmts['originalLeaseTermNumber'] * dtPmts['reportingPeriodScheduledPaymentAmount'] + dtPmts['contractResidualValue']
    print('Added field: originalLoanAmount ... ')

    # Add original interest rate, need to adjust the contractResidual to the following to last period
    dtPmts['originalInterestRatePercentage'] = 12 * np.maximum(np.rate(dtPmts['originalLeaseTermNumber'],dtPmts['reportingPeriodScheduledPaymentAmount'],
                                                        -dtPmts['acquisitionCost'],dtPmts['contractResidualValue'],1),0)
    dtPmts['nextInterestRatePercentage'] = dtPmts['originalInterestRatePercentage']
    print('Added field: original and nextInterestRatePercentage ... ')

    # Add vintage
    dtPmts['vintage'] = dtPmts['originationDate'].dt.year
    print('Added field: vintage ...')

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
    dtPmts['netLosses'] = np.maximum(dtPmts['chargedOffAmount'] - dtPmts['repurchaseAmount'],0).fillna(0)
    print('Added field: netLosses ...')

    #Add region
    dtRegion = read_dict('states.csv')
    dtPmts = pd.merge(dtPmts, dtRegion, how='left', left_on='lesseeGeographicLocation', right_on='state', copy=False)
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
                               'reportingPeriodActualEndBalanceAmount','actualInterestCollectedAmount',#'principalActualPrepaid',
                               'recoveredAmount','repossessedProceedsAmount'].sum()

def fit_reporting_model(dtPmts):

    dtPmts = dtPmts.rename(columns={'chargedOffAmount':'chargedoffPrincipalAmount',
                                    'liquidationProceedsAmount':'repossessedProceedsAmount',
                                    'reportingPeriodEndingActualBalanceAmount':'reportingPeriodEndingLeaseBalanceAmount',
                                    'reportingPeriodSecuritizationValueAmount':'reportingPeriodBeginningLoanBalanceAmount',
                                    'reportingPeriodEndActualSecuritizationAmount':'reportingPeriodActualEndBalanceAmount',
                                    'coLesseePresentIndicator':	'coObligorIndicator',
                                    'paidThroughDate':	'interestPaidThroughDate',
                                    'reportingPeriodBeginDate':	'reportingPeriodBeginningDate',
                                    'reportingPeriodEndDate':	'reportingPeriodEndingDate',
                                    'scheduledTerminationDate':	'loanMaturityDate',
                                    'gracePeriod':	'gracePeriodNumber',
                                    'lesseeCreditScore':'obligorCreditScore',
                                    'lesseeEmploymentVerificationCode':'obligorEmploymentVerificationCode',
                                    'lesseeIncomeVerificationLevelCode':'obligorIncomeVerificationLevelCode',
                                    'originalLeaseTermNumber':	'originalLoanTerm',
                                    'remainingTermNumber':	'remainingTermToMaturityNumber',
                                    'lesseeCreditScoreType':	'obligorCreditScoreType',
                                    'lesseeGeographicLocation':	'obligorGeographicLocation',
                                    'primaryLeaseServicerName':	'primaryLoanServicerName'})

    return dtPmts

def describe_raw_data(dtPmts):

    fieldStr = const.booleanFields() + const.stringFields()
    dtPmts = dtPmts[fieldStr]
    secKeys = np.sort(dtPmts['securitizationKey'].unique())
    dtDesc = pd.DataFrame()

    for s in secKeys:
        print(s)
        secIndx = np.where(dtPmts['securitizationKey']==s)
        dtTemp = dtPmts.iloc[secIndx].describe(exclude=[np.number])
        dtTemp.loc['vals'] = np.nan
        dtTemp.loc['nan'] = np.nan
        for f in dtTemp.columns:
            print(f)
            dtTemp[f].loc['vals'] = dtPmts[f].unique()
            dtTemp[f].loc['nan'] = dtPmts[f].isnull().sum()
        dtTemp.index = s + ' ' + dtTemp.index
        dtDesc = pd.concat([dtDesc,dtTemp],axis=0)

    dtDesc.to_csv('dtDesc.csv')

def main(argv = sys.argv):

dtPmts = clean_ald_files(dtRaw)
dtPmts = append_calc_fields(dtPmts)
dtPmts = fit_reporting_model(dtPmts)

describe_raw_data(dtPmts)
cashflow_vetting(dtPmts).to_csv('cf.csv')
dtErrors,dtDescNum,dtDescStr = data_vetting(dtPmts)
#dtDescNum.to_csv('descnum.csv')
#dtDescStr.to_csv('descstr.csv')
dtErrors.to_csv('Error Log/cleanErrors20170508.csv')


if __name__ == "__main__":
    sys.exit(main())