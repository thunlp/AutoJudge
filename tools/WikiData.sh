#!/usr/bin/env bash
wget https://dumps.wikimedia.org/zhwiki/20180401/zhwiki-20180401-pages-articles-multistream.xml.bz2
bunzip2 zhwiki-20180401-pages-articles-multistream.xml.bz2
cat zhwiki-20180401-pages-articles-multistream.xml |python WikiExtractor.py -b600M -o extracted > vocabulary.txt
opencc -i wiki_00 -o wiki_00_chs -c zht2zhs.ini
python3 FileSeg.py wiki_00_chs