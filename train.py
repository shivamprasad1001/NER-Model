import json
import random
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

# Simple NER trainer that avoids spaCy lookup table issues
class SimpleNERTrainer:
    def __init__(self):
        """Initialize simple NER trainer"""
        self.nlp = None
        self.entity_labels = set()
        
    def load_doccano_data(self, file_path):
        """Load and parse Doccano JSONL format data"""
        training_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['text']
                labels = data['labels']
                
                # Convert to training format
                entities = []
                for label in labels:
                    start, end, entity_type = label[0], label[1], label[2]
                    entities.append((start, end, entity_type))
                    self.entity_labels.add(entity_type)
                
                training_data.append((text, {"entities": entities}))
        
        print(f"Loaded {len(training_data)} training examples")
        print(f"Entity types: {sorted(self.entity_labels)}")
        return training_data
    
    def create_minimal_model(self):
        """Create minimal spaCy model without lookup dependencies"""
        import spacy
        from spacy.lang.en import English
        
        # Create basic English model
        self.nlp = English()
        
        # Add minimal required components
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")
        
        # Add entity labels
        for label in self.entity_labels:
            ner.add_label(label)
        
        return self.nlp
    
    def train_model(self, training_data, n_iter=100, batch_size=8):
        """Train the NER model with minimal dependencies"""
        import spacy
        from spacy.training import Example
        from spacy.util import minibatch
        
        # Create model
        self.create_minimal_model()
        
        # Split data
        random.shuffle(training_data)
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Training on {len(train_data)} examples")
        print(f"Validating on {len(val_data)} examples")
        
        # Get optimizer
        optimizer = self.nlp.begin_training()
        
        # Training loop
        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Create batches
            batches = minibatch(train_data, size=batch_size)
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update model
                self.nlp.update(examples, drop=0.3, losses=losses, sgd=optimizer)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{n_iter}, Loss: {losses.get('ner', 0):.4f}")
        
        # Final evaluation
        self.evaluate_model(val_data)
        
        return self.nlp
    
    def evaluate_model(self, test_data):
        """Simple evaluation"""
        if not self.nlp:
            print("No model to evaluate!")
            return
        
        correct = 0
        total = 0
        
        for text, annotations in test_data:
            doc = self.nlp(text)
            
            # Get true entities
            true_entities = set()
            for start, end, label in annotations['entities']:
                true_entities.add((start, end, label))
            
            # Get predicted entities
            pred_entities = set()
            for ent in doc.ents:
                pred_entities.add((ent.start_char, ent.end_char, ent.label_))
            
            # Calculate metrics
            correct += len(true_entities.intersection(pred_entities))
            total += len(true_entities)
        
        if total > 0:
            accuracy = correct / total
            print(f"\\nEvaluation Results:")
            print(f"Correct predictions: {correct}")
            print(f"Total entities: {total}")
            print(f"Accuracy: {accuracy:.3f}")
        else:
            print("No entities to evaluate!")
    
    def save_model(self, output_dir):
        """Save the trained model"""
        if not self.nlp:
            print("No model to save!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.nlp.to_disk(output_path)
        
        # Save entity labels
        with open(output_path / "entity_labels.pkl", "wb") as f:
            pickle.dump(list(self.entity_labels), f)
        
        print(f"Model saved to {output_path}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        import spacy
        
        self.nlp = spacy.load(model_path)
        
        # Load entity labels
        with open(Path(model_path) / "entity_labels.pkl", "rb") as f:
            self.entity_labels = set(pickle.load(f))
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, text):
        """Make predictions on new text"""
        if not self.nlp:
            print("No model loaded!")
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
        
        return entities
    
    def interactive_test(self):
        """Interactive testing"""
        if not self.nlp:
            print("No model loaded!")
            return
        
        print("\\n=== Interactive NER Testing ===")
        print("Enter text to test (type 'quit' to exit):")
        
        while True:
            text = input("\\n> ")
            if text.lower() == 'quit':
                break
            
            entities = self.predict(text)
            
            if entities:
                print("\\nEntities found:")
                for ent in entities:
                    print(f"  '{ent['text']}' -> {ent['label']} ({ent['start']}-{ent['end']})")
            else:
                print("No entities found.")

def main():
    """Main function - simple training pipeline"""
    
    # Check if spaCy is available
    try:
        import spacy
        print("✓ spaCy is available")
    except ImportError:
        print("✗ spaCy not found. Install with: pip install spacy")
        return
    
    # Initialize trainer
    trainer = SimpleNERTrainer()
    
    # Load data
    try:
        print("\\nLoading training data...")
        training_data = trainer.load_doccano_data("ner_training_data.jsonl")
        
        if not training_data:
            print("No training data found!")
            return
            
    except FileNotFoundError:
        print("Training data file 'ner_training_data.jsonl' not found!")
        print("Please make sure your JSONL file is in the current directory.")
        return
    
    # Train model
    print("\\nStarting training...")
    try:
        trainer.train_model(training_data, n_iter=50, batch_size=4)
        
        # Save model
        trainer.save_model("./simple_ner_model")
        
        # Interactive testing
        trainer.interactive_test()
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("\\nTrying alternative approach...")
        
        # Fallback: provide manual testing
        print("\\nManual testing mode:")
        print("The model training encountered issues, but you can still test with sample data")

if __name__ == "__main__":
    print("=== Simple NER Trainer ===")
    print("This version avoids spaCy lookup table issues")
    print("\\nRequired: pip install spacy")
    print("\\nStarting...\\n")
    
    main()