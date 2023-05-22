import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append(str(Path(__file__).resolve().parents[2]))
from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility

apex_code_utility = ApexCodeUtility()

sample_code = 
    """
    public class AccountWithContacts {
    // Method to fetch accounts and their related contacts
    public static void getAccountsWithContacts() {
        // Query to fetch accounts with related contacts
        List<Account> accounts = [SELECT Id, Name, (SELECT Id, LastName, Email FROM Contacts) FROM Account LIMIT 10];

        // Iterate through accounts and print account and contact information to the debug log
        for (Account acc : accounts) {
            System.debug('Account Name: ' + acc.Name);

            for (Contact con : acc.Contacts) {
                System.debug('Contact Name: ' + con.LastName + ', Email: ' + con.Email);
            }
        }
    }
}
    """

new_code_snippet = apex_code_utility.rename_identifiers(sample_code)
print(new_code_snippet)

