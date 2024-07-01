
if [ $# -eq 0 ]; then
  echo "Please provide the destination folder as an argument."
  exit 1
fi

dest_folder="$1"
echo $dest_folder
cd "$dest_folder"

wget -N https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip

unzip region_descriptions.json.zip
rm -rf region_descriptions.json.zip