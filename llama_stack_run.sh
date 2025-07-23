#!/bin/bash
echo "-----starting to llama-stack docker now---------"
pip install llama-stack-haystack
pip install gradio
echo "starting the llama-stack server in the background"
llama stack run starter &
echo "waiting for llama-stack server to be available on http://localhost:8321/"
while ! curl -s http://localhost:8321/ > /dev/null; do
  echo "Server not ready yet, waiting..."
  sleep 3
done
echo "Server is up and running!"
echo "---------running the RAG app--------------"
python /root/demo.py
sleep 999999
echo "finished everything"
