import sys
from data_processing import load_data, rename_target, split_dataset
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QComboBox, QLabel, QLineEdit, QFileDialog)

from model_training import train_krr_model, generate_predictions


class MLGUIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.dataframe = None
        self.columns = []
        self.parameter_inputs = {}  # 存储参数输入框引用
        self.X_train, self.X_test, self.y_train, self.y_test, self.model = None, None, None, None, None
        self.train_results = None
        self.test_results = None

    def initUI(self):
        self.setWindowTitle('Machine Learning GUI')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        centralWidget.setLayout(layout)

        # Upload CSV Button
        self.btn_upload_csv = QPushButton('Upload Your Dataset', self)
        self.btn_upload_csv.clicked.connect(self.upload_dataset)
        layout.addWidget(self.btn_upload_csv)

        # Column selector for the target variable
        self.column_selector = QComboBox(self)
        self.column_selector.currentIndexChanged.connect(self.update_target_column)
        layout.addWidget(QLabel('Select Target Column:'))
        layout.addWidget(self.column_selector)

        # Test set ratio input
        self.test_ratio_input = QLineEdit(self)
        layout.addWidget(QLabel('Test Set Ratio (0-1):'))
        layout.addWidget(self.test_ratio_input)

        # Add additional GUI components here
        self.split_button = QPushButton('Split Data', self)
        self.split_button.clicked.connect(self.split_data)
        layout.addWidget(self.split_button)

        # Model selection
        self.model_selection = QComboBox(self)
        self.model_selection.addItems(['Select a model', 'KRR', 'Random Forest', 'XGBoost'])
        self.model_selection.currentIndexChanged.connect(self.model_changed)
        layout.addWidget(self.model_selection)

        # Placeholder for parameter inputs (this will be dynamic)
        self.params_layout = QVBoxLayout()
        layout.addLayout(self.params_layout)

        self.train_button = QPushButton('Train Model', self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Buttons for predictions and report
        self.btn_predict = QPushButton('Generate Predictions', self)
        self.btn_predict.clicked.connect(self.generate_predictions)
        layout.addWidget(self.btn_predict)

        self.btn_save = QPushButton('Save Results', self)
        self.btn_save.clicked.connect(self.save_results)
        layout.addWidget(self.btn_save)

        self.btn_report = QPushButton('Generate Report', self)
        self.btn_report.clicked.connect(self.generate_report)
        layout.addWidget(self.btn_report)

    def upload_dataset(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home', "CSV files (*.csv);;Excel files (*.xlsx)")
        if fname:
            try:
                self.dataframe, self.columns = load_data(fname)
                self.update_column_selector()  # Assume this updates the UI with the columns
                print("Data loaded successfully.")
                print("Columns:", self.columns)
            except ValueError as e:
                print(e)

    def update_target_column(self):
        if self.dataframe is not None:
            selected_column = self.column_selector.currentText()
            if selected_column:
                self.dataframe = rename_target(self.dataframe, selected_column)

    def update_column_selector(self):
        self.column_selector.clear()  # 清除现有的选项
        if self.dataframe is not None:
            self.column_selector.addItems(self.dataframe.columns.tolist())  # 确保列名被转换为列表

    def split_data(self):
        test_size = float(self.test_ratio_input.text())  # Assuming you have a QLineEdit for test_size input
        target_column = 'target'  # Assuming target column is always named 'target'
        if self.dataframe is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(self.dataframe, target_column, test_size)
            print("Data split into training and testing sets.")
            # You might want to do something with the split data here, like displaying them in the GUI

    def model_changed(self, index):
        # Clear existing parameter inputs
        for i in reversed(range(self.params_layout.count())):
            widget_to_remove = self.params_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)

        # Add parameter inputs based on selected model
        model_params = {
            'KRR': ['Alpha', 'Gamma', 'Kernel', 'Degree', 'Coef0'],
            'Random Forest': ['n_estimators', 'max_depth', 'min_samples_split'],
            'XGBoost': ['n_estimators', 'max_depth', 'learning_rate']
        }
        selected_model = self.model_selection.currentText()
        if selected_model in model_params:
            for param in model_params[selected_model]:
                self.add_parameter_input(param)

    def add_parameter_input(self, name):
        line_edit = QLineEdit(self)
        self.params_layout.addWidget(QLabel(f'{name}:'))
        self.params_layout.addWidget(line_edit)
        self.parameter_inputs[name] = line_edit  # 存储输入框引用

    def train_model(self):
        if self.X_train is not None and self.y_train is not None:
            params = {name: self.parameter_inputs[name].text() for name in self.parameter_inputs}
            if self.model_selection.currentText() == 'KRR':
                # Convert parameters to correct type
                params = {
                    'alpha': float(params.get('Alpha', 1.0)),
                    'gamma': float(params.get('Gamma', None)) if params.get('Gamma') else None,
                    'kernel': params.get('kernel', 'linear'),
                    'degree': int(params.get('Degree', 3)),
                    'coef0': float(params.get('Coef0', 1))
                }
                self.model = train_krr_model(self.X_train, self.y_train, params)
                print("KRR model trained.")
                # Additional code to handle the trained model, display results, or evaluate model

    def generate_predictions(self):
        if self.model and self.X_train is not None:
            self.train_results, self.test_results = generate_predictions(self.model, self.X_train, self.X_test, self.y_train, self.y_test)
            print("Predictions generated.")
        else:
            print("Model not trained or data not available.")

    def save_results(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            if not fileName.endswith('.csv'):
                fileName += '.csv'
            self.train_results.to_csv(fileName.replace('.csv', '_train.csv'), index=False)
            self.test_results.to_csv(fileName.replace('.csv', '_test.csv'), index=False)
            print("Files saved:", fileName.replace('.csv', '_train.csv'), "and", fileName.replace('.csv', '_test.csv'))

    def generate_report(self):
        print("Generating report...")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MLGUIApp()
    ex.show()
    sys.exit(app.exec_())
