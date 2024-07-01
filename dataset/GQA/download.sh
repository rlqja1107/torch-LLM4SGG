
if [ $# -eq 0 ]; then
  echo "Please provide the destination folder as an argument."
  exit 1
fi

dest_folder="$1"
echo $dest_folder
cd "$dest_folder"


wget -N https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
rm -rf images.zip


if ! test -f "GQA_200_ID_Info.json"; then
  wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eDZfLqV5sPteIWLxb432HwlMTXmeq4CI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eDZfLqV5sPteIWLxb432HwlMTXmeq4CI" -O GQA_200_ID_Info.json && rm -rf ~/cookies.txt
fi

if ! test -f "GQA_200_Test.json"; then
  wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nS0jDbQf73aWbjtyQ_sZyBoAHDucPn8o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nS0jDbQf73aWbjtyQ_sZyBoAHDucPn8o" -O GQA_200_Test.json && rm -rf ~/cookies.txt
fi

if ! test -f "GQA_200_Train.json"; then
  wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-QQ4PVYIKsDq7An9VRj2AXRQFgdIkGxC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-QQ4PVYIKsDq7An9VRj2AXRQFgdIkGxC" -O GQA_200_Train.json && rm -rf ~/cookies.txt
fi
