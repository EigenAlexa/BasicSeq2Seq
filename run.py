from seq_model import SeqModel
import tensorflow as tf

def main():

    hyperparams = {
    }
    # TODO check if model already exists and load it
    with tf.Session() as sess:
        model = SeqModel(sess=sess)
        load_from_file = input("Would you like to load from file?y/n ")
        if load_from_file == 'y':
            model.load()
        else: 
            print("Train model")
            model.train()
            print("Saving model")
            model.save()
        print("START TESTING")
        print("'Hi I'm Eigen. Let's talk'")
        while True:
            user_response = input()
            model_response = model.feed(user_response.lower())
            print(model_response)
if __name__ == "__main__":
    main()

