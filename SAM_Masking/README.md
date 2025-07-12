**1.	Downloads and Installation**

For Windows users:

PuTTY: https://www.putty.org/
WinSCP: https://winscp.net/eng/index.php?

Download `SAM_Masking` folder from GitHub: https://github.com/Jamie-Dean-Lab/normal-brain-toxicity

Install the libraries in requirements.txt via Python: `pip install -r requirements.txt`

<br/>

**2.	Cluster**

YOU MUST BE CONNECTED TO UCL WIFI OR THE UCL VPN IN ORDER TO USE THE CLUSTER: https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
- I would personally recommed using Cisco to connect to the UCL VPN.

Once you have access to the cluster, you can ‘log in’ to the cluster on PuTTY and WinSCP. 
- This requires the IP address and password. The login details are the **same** for both PuTTY and WinSCP.

<br/>

PuTTY: 

- On PuTTY, type in the IP address in the 'Host Name (or IP address)' field. Leave the rest of the settings as they are.
  
  ![image](https://github.com/user-attachments/assets/5aec4a6c-6017-4dc7-a443-9f0c518fcdc7)
- You will then be asked to enter the password:

  ![image](https://github.com/user-attachments/assets/fb78044c-39da-45ca-b157-99689b2dc63e)
- Once you enter the password, you will have access to the command line.

<br/>

WinSCP:

- On WinSCP, a 'Login' dialog should appear on your screen.
- On the left hand side will be a white box. Inside this white box, press 'New Site'.
- Enter the IP address in the 'Host name:' field. No need to change anything else. You can ignore the 'User name' and 'Password' fields.
  
  ![image](https://github.com/user-attachments/assets/cdfa9f72-c87f-454d-8c9a-5ef4f68cc6a1)
- You will then be asked to enter the password.
- Once you enter your password, you will have access to the files on the cluster.

<br/>

PuTTY will be your (Linux) command line to run your code.

WinSCP will be your GUI to make it easier to see your files.
  This is recommended, but optional – if you are comfortable using the Linux CL to navigate, move and download files, you may not need WinSCP. 

Create a new folder in the cluster workspace and copy your SAM_Masking folder into this folder.
  Let’s call this new folder `Name`.

<br/>  

**3.	Histology Data**

This is where you will store your downloaded patient data.

Data available at: https://rodare.hzdr.de/record/1849

<br/>

In `Name`, you should currently only have one folder called `SAM_Masking`.
Create a new folder within `Name` called `patient_data`.

<br/>

For now, we are interested in the Histology split zip files.

- Focus on one mouse at a time.
- Download each split zip for that particular mouse (e.g. P2A_B6_M1_Histology.zip.001).
- Copy this into `patient_data` in the cluster workspace.
- Delete it from your local device.
- Repeat for each split zip (P2A_B6_M1_Histology.zip.002, P2A_B6_M1_Histology.zip.003, etc.).
- Once they’re all in `patient_data`, you can organise the files.

It will be easier if you follow the same organisation system that the code currently deals with. You are welcome to change this, but this would involve changing the code, the command line arguments, and the config file (more on this later).

<br/>

Within `patient_data`:

![image](https://github.com/user-attachments/assets/af7941c5-a2b0-46af-ac07-74144eb39963)

<br/>

And store your hist images within each slice (i.e. within `P2A_C3H_M5_0001_1`):

![image](https://github.com/user-attachments/assets/0c8c90a7-a876-40e8-b7ef-6793cf8f2b05)

<br/>

**4.	Config file**

Within the `SAM_Masking` folder is a config file – open this and make the necessary changes to the directories / whatever else you think needs changing.

<br/>

**5.	Run SAM**

Run your code via the CL in PuTTY. Ensure that you are working in the correct directory before running anything.

`nohup /usr/bin/python3.10 Inference_SAM.py –config "/home/dgs/Saarah/config_infer_CRC01.ini" --gpus 1 --patient "/home/dgs/Saarah/patient_data/P2A_C3H_M5_0002_1"`

Change this command as required – you will need to change the directories!

Make sure you understand what is going on in this command, including the use of `nohup`.

<br/>

**5.  Track of Mice**
https://docs.google.com/spreadsheets/d/1kYST8hiWFySvItZ7RXriuvvu7tU1koKaKIo3zB_VFFA/edit?usp=sharing

