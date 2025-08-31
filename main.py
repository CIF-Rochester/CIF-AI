import os

import argparse
import logging
import subprocess
import json
import ollama
import numpy as np
from pydantic.dataclasses import dataclass

HISTORY_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "history")
EMBED_NUM = 0
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
NUM_BEST_MATCHES = 5

def get_history_file(channel: str) -> (str, bool):
    for name in os.listdir(HISTORY_PATH):
        if name == channel + ".txt":
            return os.path.join(HISTORY_PATH, name), True
    return os.path.join(HISTORY_PATH, channel + ".txt"), False

def generate_response(messages) -> str:
    data = {"model":"nauticock", "stream":False, "keep_alive":0, "messages":messages }
    with open('chat_payload.json', 'w') as f:
        json.dump(data, f, indent=4)
    command = "curl --request POST --silent --data @chat_payload.json http://localhost:11434/api/chat | jq \'.message.content\'"
    output = subprocess.run(command, capture_output=True, shell=True)
    response = output.stdout.decode("utf-8")
    if not output.stderr and response != "null":
        return response[1:-2]
    else:
        raise Exception(output.stderr.decode("utf-8"))

def embed(prompt: str) -> np.ndarray:
    response = ollama.embed(model='nomic-embed-text', input=[prompt]).embeddings[0]
    return np.array(response)

def embed_chunk(chunks: list, file_name: str):
    responses = ollama.embed(model='nomic-embed-text', input=chunks).embeddings
    for i, chunk in enumerate(chunks):
        with open(f'references/embeddings/{file_name}-{i}.txt', 'x') as f:
            f.write(f"{chunk}\n{str(responses[i])}")
    return responses

def get_chunks(file_name: str):
    chunks = []
    with open(file_name, 'r') as f:
        content = f.read().split()
    for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(content[i:i+CHUNK_SIZE])
        chunks.append(chunk)
    return chunks

def embed_references() -> list:
    for file in os.listdir("references/embeddings"):
        os.remove("references/embeddings/" + file)
    references = []
    for file in os.listdir("references/wiki.wiki"):
        if file.endswith(".md") and file[0] != '_':
            file_chunks = get_chunks("references/wiki.wiki/" + file)
            chunk_embeddings = embed_chunk(file_chunks, file[:-3])
            for i, embedding in enumerate(chunk_embeddings):
                references.append([file_chunks[i], np.array(embedding)])
    return references

def get_embeddings() -> list:
    references = []
    for file in os.listdir("references/embeddings"):
        with open("references/embeddings" + file, 'r') as f:
            contents = f.read().split("\n")
        references.append([contents[0], np.fromstring(contents[1][1:-1], dtype=float, sep=',')])
    return references


def get_closest_chunks(prompt: str, references: list) -> list:
    prompt_embedding = embed(prompt)
    pm = np.linalg.norm(prompt_embedding)
    def compare(item1, item2):
        m1 = np.linalg.norm(item1[1]) * pm
        m2 = np.linalg.norm(item2[1]) * pm
        a1 = np.dot(item1[1], prompt_embedding) / m1
        a2 = np.dot(item2[1], prompt_embedding) / m2
        if a1 < a2:
            return 1
        elif a1 == a2:
            return 0
        else:
            return -1
    references.sort(key=compare)
    return references[:NUM_BEST_MATCHES]

def create_directories():
    if not os.path.exists("references"):
        os.mkdir("references")
    if not os.path.exists("references/embeddings"):
        os.mkdir("references/embeddings")
    if not os.path.exists("references/wiki.wiki"):
        subprocess.run(['git','clone','https://github.com/CIF-Rochester/wiki.wiki.git'], cwd='references', shell=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Ollama History Parser")
    parser.add_argument("--user", "-u", help="Username of user", default='')
    parser.add_argument("--prompt", "-p", help="Prompt", default='')
    parser.add_argument("--channel", "-c", help="Channel of prompt", default='')
    parser.add_argument("--update_embeddings", "-w", default=True, help="Pulls latest constitution and wiki info and embeds them")


    args = parser.parse_args()

    create_directories()
    if args.update_embeddings:
        subprocess.Popen('git pull', cwd='references/wiki.wiki')
        references = embed_references()
    else:
        references = get_embeddings()
    if args.prompt:
        prompt = args.prompt.replace('\n', '\\n').replace('\"', '\\\"').replace('\'', '\\\'')
        best_matches = get_closest_chunks(prompt, references)
        for match in best_matches:
            print(match[0])
        filename, exists = get_history_file(args.channel)
        message_object = {"role":"user", "content":f"{args.user} says: {prompt}"}
        if exists:
            messages = []
            with open(filename) as f:
                content = f.read().split("\n")
                for i, line in enumerate(content):
                    if i % 2 == 0:
                        messages.append({ "role":"user", "content":f"{args.user} says: {line}" })
                    else:
                        messages.append({ "role":"assistant", "content":f"{line}" })
            messages.append(message_object)
        else:
            messages = [message_object]
        response = generate_response(messages)
        if not response is None and len(response.strip()) > 0:
            print(response)
            response = response.replace('\n', '\\n').replace('\"', '\\\"').replace('\'', '\\\'')
            if exists:
                with open(filename, 'a') as f:
                    f.write(f"\n{prompt}")
            else:
                with open(filename, 'x') as f:
                    f.write(prompt)
            with open(filename, 'a') as f:
                f.write(f"\n{response}")
        else:
            raise Exception("No response generated")
