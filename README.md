SortOUISIR.py is used to generated the test dataset. First email gaitdb_admin@am.sanken.osaka-u.ac.jp the signed relise agreement form OUMVLP_ReleaseAgreement.pdf. After a few days they will email you a password that can be used to unzip the datafolders that can be downloaded from http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html. Create a home folder. Place the data in a folder named 'Data', this data folder should contain all angle folders. In the home folder create a folder named 'Code' and place the SortOUISIR.py script inside, also in the home folder create a project folder named, for example, 'Gate_sis'. In this folder place a 'Code', 'data', 'weights' folder. Place the RCNNSiame.py script in the 'Code' folder.

The environment should look like this:

HOME
-Code
--SortOUISIR.py
-Data
--all angles
-'Project name'
--Code
---RCNNSiame.py
---other models that youd like to test
--data
--weights


'Edit the SortOUISIR.py script to include your project name where 'Gate_sis' is present'

The script can be opened and run using python 3.5

While the RCNNSiame.py script needs keras and a tensorflow-gpu environment to run. Idealy use an ananconda virtual environment.
