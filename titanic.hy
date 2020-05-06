;; First, let's import some libraries

(import re
        [pandas :as pd]
        [keras.utils :as k-utils]
        [matplotlib.pyplot :as plt]
        [sklearn.model_selection [train_test_split]]
        [tensorflow.keras.models [Sequential]]
        [tensorflow.keras.layers [Dense BatchNormalization]])

;; Data processing
;; Feature engineering
(defn add-feature
  [ds feature-name data]
  "Adds/replace a column by name.
ds: data-set to add/change the feature to.
feature-name: name of the column to add/modify.
data: the data series to add."
  (setv (get ds feature-name) data)
  ds)

(defn extract-from-text
  [column regexstr]
  "Extracts data from text based on a regex string."
  (defn find-title [name]
    (-> (re.findall regexstr name)
       first
       .strip))
  (.map column find-title))

;; Extract the person's title by the name.
(defn extract-titles [column]
  "Just partially applies extract-from-text"
  (extract-from-text column "\s(\w+)\."))

(defn add-titles [ds]
  "Adds title information to the data-set based on name."
  (add-feature ds "Title" (extract-titles ds.Name)))

(defn family-size [ds]
  "Create a new feature for family size of the person."
  (+ (get ds "SibSp") (get ds "Parch")))

(defn create-alone [ds]
  "Classify people as alone or not."
  (add-feature ds
               "Alone"
               (pd.Series (map (fn [e] (if (> e 0) e 1)) (family-size ds)))))

;; Cabin has too many NaNs, it must go
;; PassengerId has no information at all, same for ticket
(defn drop-useless
  [ds &optional [to-drop ["PassengerId" "Ticket" "Name" "Cabin"]]]
  "Drops unused columns."
  (get ds (.drop ds.columns to-drop)))

;; Input missing data
(defn input-by-mean [ds feature]
  "Fills missing data using the mean."
  (setv feature-mean (.mean (get ds feature)))
  (add-feature ds feature (-> (get ds feature)
                             (.fillna feature-mean))))

;; Get one-hot encoded feature
(defn get-dummies [ds name]
  (.drop (pd.concat [ds
                     (pd.get-dummies (get ds name)
                                     :prefix name
                                     :drop-first True)]
                    :axis 1)
         name
         :axis 1))

;; So now, we create a pipeline for data processing:
(defn process-data [ds]
  "Main data processing pipeline."
  (-> (.copy ds)
     (add-feature "Family_size" (family-size ds))
     create-alone
     add-titles
     drop-useless
     (get-dummies "Title")
     (get-dummies "Pclass")
     (get-dummies "Sex")
     (get-dummies "Embarked")
     (input-by-mean "Age")))

;; Prepare the data for the algorithm
(defn split-train-validation [ds]
  "Splits a data-set between Train, test, validation train and validation test."
  ;; get the training set
  (setv X (-> (.copy ds)
             (drop-useless ["Survived"])))
  ;; get the validation columns
  (setv y (-> (.copy ds)
             (get "Survived")
             k-utils.to-categorical))
  (train-test-split X.values y :test-size 0.2 :random-state 999 :stratify y))

(defn train-model
  [model X-train X-test y-train y-test &optional [epochs 10] [verbose 1]]
  "Trains the machine learning model."
  (setv history (.fit model
                      X-train y-train
                      :validation_data (, X-test y-test)
                      :epochs epochs
                      :verbose verbose))
  history)

(defn plot-accuracy [history]
  "Just plots a graph of accuracy given a history object."
  (setv acc (get history.history "acc")
        ;; validation accuracy
        val-acc (get history.history "val_acc"))
  (plt.plot acc)
  (plt.plot val-acc)
  (plt.ylabel "Accuracy")
  (plt.xlabel "Epoch")
  (plt.grid)
  (plt.show))

(defn join-train-test [train test]
  ;; How to solve mismatch in train and test set after categorical encoding?
  ;; https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f
  (setv (get train "TRAIN") 1)
  (setv (get test "TRAIN") 0)
  (pd.concat [train test]))

(defn separate-train-test [ds]
  "Separates train and test data that were put together by join-train-test."
  (defn select [train-or-test]
    (.drop (get ds (= train-or-test (get ds "TRAIN"))) ["TRAIN"] :axis 1))
  (setv train (select 1))
  (setv test (.drop (select 0) ["Survived"] :axis 1))
  (, train test))

;; The builder for the model
(defn build-model [input-size]
  "A description for the neural network model."
  (setv model (Sequential))
  (.add model (Dense 9 :input-dim input-size :activation "relu"))
  (.add model (BatchNormalization))
  (.add model (Dense 9 :activation "relu"))
  (.add model (Dense 9 :activation "relu"))
  (.add model (Dense 9 :activation "relu"))
  (.add model (Dense 2 :activation "softmax"))
  (.compile model :optimizer "adam"
            :loss "categorical_crossentropy"
            :metrics ["accuracy"])
  model)

(defn main []
  "The main function that processes, trains, tests the model and displays the results."
  (setv test-file (pd.read-csv "test.csv"))
  (setv [train test] (-> (join-train-test (pd.read-csv "train.csv")
                                         test-file)
                        process-data
                        separate-train-test))
  (setv [X-train X-val y-train y-val]
        (split-train-validation train))
  (setv model (build-model (second X-train.shape)))
  (setv history (train-model model X-train X-val y-train y-val :epochs 500))
  (setv predictions (.predict-classes model test))
  (setv result (pd.DataFrame { "PassengerId" (get test-file "PassengerId")
                              "Survived" predictions}))
  (result.to-csv "results.csv"
                 :encoding "utf-8"
                 :index False)
  (plot-accuracy history))

(main)
