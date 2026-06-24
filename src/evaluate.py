import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def evaluate_model(model, x_test, y_test, model_name="Model"):
    y_pred = model.predict(x_test)

        total = len(y_pred)
            correct = sum(y_pred == y_test)
                incorrect = sum(y_pred != y_test)
                    accuracy = correct * 100 / total

                        print(f"\n--- {model_name} Results ---")
                            print("Total Prediction :", total)
                                print("Correct Prediction :", correct)
                                    print("Incorrect Prediction :", incorrect)
                                        print("Accuracy :", accuracy, "%")

                                            return y_pred


                                            def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
                                                cm = confusion_matrix(y_test, y_pred)
                                                    print(cm)

                                                        fig = plt.figure(figsize=(10, 6))
                                                            plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
                                                                plt.title(title + "\n")
                                                                    plt.colorbar()
                                                                        plt.ylabel("Expected Label")
                                                                            plt.ylabel("Predicted Label")
                                                                                plt.show()
                                                                                
