# To Start I had to get the files off of my laptop onto the github. Due to the files being to large gitlfs was used.
# Go to the local project folder
# cd C:\Users\insur\Downloads\SOSC

# Connect the local folder to the GitHub repository
git remote set-url origin https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git

# Enable Git Large File Storage because some of the datasets
# are too large to upload through normal GitHub storage
# git lfs install

# Tell Git LFS to track Excel and CSV data files
# git lfs track "*.xls"
# git lfs track "*.csv"

# Add the Git LFS configuration file
# git add .gitattributes

# Add the Harvard General Inquirer dictionary
# git add inquireraugmented.xls

# Add the World Bank economic data
# git add world_bank_economic_data.csv

# Commit the files to the local Git repository
# git commit -m "Initial files September 1st"

# Upload the committed files to the GitHub repository
# git push -u origin main

# Given that this was done from my local file, this step will not need to be repeated for people using this repo.