from utils.functions import *
import joblib


root_dir = os.path.dirname(os.path.abspath(__file__))

def main():
        # Load and prepare data
    X_train, X_test, y_train, y_test, scaler,features = load_and_prepare_data()
    # Train MLP model
    mlp_model = build_and_train_mlp(X_train, X_test, y_train, y_test)

    # Train ANN model
    ann_model = build_and_train_ann(X_train, y_train)

    # Save the scaler and ANN model
    joblib.dump(scaler, os.path.join(root_dir, "model", "scaler.pkl"))
    ann_model.save(os.path.join(root_dir, "model",'breast_cancer.h5'))


if __name__ == "__main__":
    main()
