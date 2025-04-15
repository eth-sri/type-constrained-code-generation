use oxc_allocator::Allocator;
use oxc_parser::{Parser, ParserReturn};
use oxc_span::SourceType;
use std::env;
use std::fs;

fn main() {
    // Get the file path from the command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file_path>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];

    // Read the file content
    let source_text = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(err) => {
            eprintln!("Error reading file {}: {}", file_path, err);
            std::process::exit(1);
        }
    };

    // Memory arena where AST nodes get stored
    let allocator = Allocator::default();
    // Infer the source type based on the file extension
    let source_type = match SourceType::from_path(file_path) {
        Ok(st) => st,
        Err(x) => {
            eprintln!("Could not determine source type from file extension. {x}");
            std::process::exit(1);
        }
    };

    let ParserReturn {
        errors,   // Syntax errors
        panicked, // Parser encountered an error it couldn't recover from
        ..
    } = Parser::new(&allocator, &source_text, source_type).parse();

    if panicked || !errors.is_empty() {
            // Print errors or panic details
        if panicked {
            eprintln!("Parser panicked.");
        }

        if !errors.is_empty() {
            eprintln!("Parsing failed with the following errors:");
            for error in errors {
                eprintln!("{:?}", error);
            }
        }
        std::process::exit(1);
    }

    println!("File parsed successfully.");
    std::process::exit(0);
}
