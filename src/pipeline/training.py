from src.models.modelling import ModelingPipeline
from src.logger import configure_logger

logging = configure_logger()                  #45


class TrainingPipeline:
    def __init__(self):
        self.modeling_pipeline = ModelingPipeline()

    def train(self):
        logging.info("Starting training pipeline...")

        self.modeling_pipeline.run()

        logging.info("Training pipeline completed successfully.")
        
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.train()