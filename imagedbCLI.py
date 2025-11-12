from ImageDB import ImageDB
import argparse

db = ImageDB()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ImageDB CLI")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index images into the database")
    index_parser.add_argument("paths", nargs="+", help="Paths of images or directories containing images to index")

    query_parser = subparsers.add_parser("query", help="Query the database with a text description")
    query_parser.add_argument("text", help="Text description to query the database")
    query_parser.add_argument("-k", type=int, default=5, help="Number of top results to return")

    args = parser.parse_args()

    if args.command == "index":
        db.index(*args.paths)
    elif args.command == "query":
        results = db.query_image(args.text, k=args.k)
        for img_path, score in results:
            print(f"Image: {img_path},\tScore: {score:.4f}")
    else:
        parser.print_help()