
## Traffic Demand Prediction

Cities worldwide are increasingly turning to AI-powered solutions to tackle the issue of traffic congestion. This problem disrupts the smooth flow of transportation and poses a significant barrier to economic growth. To address this challenge effectively, the first step is to understand travel demand and patterns within urban areas comprehensively. By harnessing the power of AI, cities, and regions aim to gather critical insights into transportation dynamics. This will enable them to implement data-driven strategies and solutions to alleviate traffic congestion and promote more efficient mobility. Ultimately, this endeavor will foster economic development and prosperity.

### Task
Design a system that helps us provide valuable insights into passenger travel patterns, booking behavior, and trip cancellations, which can be used for various analyses and predict demand in the travel industry.

### Dataset Description
The dataset folder contains the following files:
* `train.csv`: 77299 x 11
* `test.csv`: 41778 x 10
* `sample_submission.csv`: 5 x 2

### Variable Description
The columns provided in the dataset are as follows:

| Column name | Description |
| :--- | :--- |
| **Index** | Represents the unique identification of datapoint |
| **geohash** | Represents geographic information regarding a place |
| **day** | Represents the day when the information is recorded |
| **timestamp** | Represents the timestamp of the record inserted in the system |
| **RoadType** | Represents the type of road in the nearby location |
| **NumberofLanes** | Represents the number of roads/lanes present in the location |
| **LargeVehicles** | Represents whether large vehicles are permitted on the specific roads/lanes |
| **Landmarks** | Represents whether there are any landmarks near the location |
| **Temperature** | Represents the temperature of the place |
| **Weather** | Represents the weather of the place |
| **demand** | Represents the demand of the traffic at the timestamp |

### Evaluation Metric
`score = max(0, 100 * (metrics.r2_score(actual, predicted)))`

### Result Submission Guidelines
* The index is "Index" and the target is the "demand" column.
* The submission file must be submitted in `.csv` format only.
* The size of this submission file must be 41778 x 2.

Ensure that your submission file contains the following:
* Correct index values as per the test file
* Correct names of columns as provided in the `sample_submission.csv` file

### Instructions
1. Click **Download dataset** to download the dataset.
2. Solve the problem in your local environment.
3. Save the predictions in a `.csv` file.
4. Click **Upload File** (under the Upload File section) to upload your prediction file (`.csv`).
5. Click **Upload File** (under the Upload Source Code section) to upload your `.ipynb` file along with any presentation file.
6. Add any instructions or comments in the Your Answer section.
7. Click **Submit**.

### Upload Source Files
You need to submit a zip or tar archive consisting of a text file explaining your approach, details about feature engineering, tools you used and the relevant source files.
