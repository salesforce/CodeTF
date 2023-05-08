import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append(str(Path(__file__).resolve().parents[2]))
from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility


def main():
    apex_code_utility = ApexCodeUtility()

    sample_code = """
        /*------------------------------------------------------------
        Author:        Sagar Avate
        Company:       Salesforce
        Description:   Loan application provider helper class.
        Inputs:        
        Test Class:     

        History
        18-01-2017  Sagar Avate        Initial Release
        05-02-2021  Vanshita Agrawal    CCB_314: Photo Update build


        ------------------------------------------------------------*/
        public without sharing class LoanApplicationProvider {
            
            //101 SOQL Fixes By GD Start 
            private static Map<Id,Loan_Application__c> loanApplicationMap;
            //101 SOQL Fixes By GD End
            //defaultFieldList contains default fields (in comma separated string form) to be queried from loan application
            //Do not add any fields here without Architect's review
            public static String defaultFieldList ='Stage__c,Sub_Stage__c,Id';//TI-457 | Varsha | DB performance

        //defaultFieldList contains default fields (in comma separated string form) to be queried from loan application
            //Do not add any fields here without Architect's review
            //public static String defaultFieldList ='Stage__c,Sub_Stage__c,Id';//TI-457 | Varsha | DB performance

            //code fix for the Re Trigger Information child record query.
            private static Map<Id,List<Re_Trigger_Information__c>> retriggerInformationMap;
        
            public static List<Re_Trigger_Information__c> getRetriggerInformation(String loanApplicationId){
                if(retriggerInformationMap == null  ){   
                    retriggerInformationMap= new Map<Id,List<Re_Trigger_Information__c>>();
                }
                if(retriggerInformationMap.containsKey(loanApplicationId)){
                    return retriggerInformationMap.get(loanApplicationId);
                }
                List<Re_Trigger_Information__c > retriggerInfoList = [SELECT Field_Name__c,Field_Value__c,Id,Section__c,IsLatestRecord__c,Loan_Application__c, Applicant__c, Field_API_Name__c, Is_Address_Modified_SPOC__c, Is_EmpDetails_Modified_By_SPOC__c, Unique_Id__c,Applicant__r.Business_Unit__c,Field_Old_Value__c FROM Re_Trigger_Information__c WHERE
                                                                            Loan_Application__c =: loanApplicationId And Active__c=true];
                
                // if(retriggerInfoList.isEmpty()) throw new InvalidDataException('Failed to provide application. Invalid application id.');  
                
                retriggerInformationMap.put(loanApplicationId,retriggerInfoList );
                //101 SOQL Fixes By GD End
                return retriggerInfoList;                                                         
            }
            }
    """
    

    # for sample_code in sample_codes:
    new_code_snippet = apex_code_utility.remove_comments(sample_code)
    print(new_code_snippet)


if __name__ == "__main__":
    main()