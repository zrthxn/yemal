# Dependencies
sudo apt install -y unzip

mkdir _data
cd _data

# Crypto RNN
echo "Crypto RNN Dataset"
curl https://pythonprogramming.net/static/downloads/machine-learning-data/crypto_data.zip > crypto_data.zip
unzip crypto_data.zip
echo "DONE"

# Cats Dogs CNN
echo "Cats Dogs CNN Dataset"
curl https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip > kagglecatsanddogs.zip
unzip kagglecatsanddogs.zip
echo "DONE"
