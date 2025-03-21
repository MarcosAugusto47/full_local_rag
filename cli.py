import argparse

from local_rag import LocalRAG


def main():
    parser = argparse.ArgumentParser(description="Local RAG CLI")
    parser.add_argument("--query", type=str, help="Query to ask the RAG system")

    args = parser.parse_args()
    rag = LocalRAG()

    if args.query:
        response = rag.get_rag_response(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {response}")
    else:
        # Interactive mode
        print("Enter your questions (type 'exit' to quit):")
        while True:
            query = input("\nQuestion: ")
            if query.lower() == "exit":
                break
            response = rag.get_rag_response(query)
            print(f"Answer: {response}")


if __name__ == "__main__":
    main()
