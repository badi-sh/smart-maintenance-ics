!pip install gradio --quiet

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gradio as gr
from pathlib import Path
import io
import os

# Set up directories for saving plots and models
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Training function
def train_model(training_csv):
    if training_csv is None:
        return "Please upload a training CSV file.", None, None, None, None, None, None

    # Load training data
    data = pd.read_csv(training_csv)
    data = data.drop(["failure", "failure_type"], axis=1)

    # Define features and target
    X = data.drop("failure_transition", axis=1)
    y = data["failure_transition"]

    # Handle missing or invalid data
    X = X.fillna(X.mean())
    y = y.fillna(y.mode()[0])

    # Ensure all features are numerical
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, model_dir / 'scaler.pkl')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_dir / 'random_forest_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)

    # Classification report
    clf_report = classification_report(y_test, y_pred)

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_path = output_dir / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()

    # 2. Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    feature_importance_path = output_dir / 'feature_importance.png'
    plt.savefig(feature_importance_path)
    plt.close()

    # 3. Learning Curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy')
    plt.plot(train_sizes, val_mean, label='Validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    learning_curves_path = output_dir / 'learning_curves.png'
    plt.savefig(learning_curves_path)
    plt.close()

    # 4. Class Distribution Bar Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y, order=y.value_counts().index)
    plt.title('Class Distribution in failure_transition')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    class_distribution_path = output_dir / 'class_distribution.png'
    plt.savefig(class_distribution_path)
    plt.close()

    return (clf_report, str(confusion_matrix_path), str(feature_importance_path),
            str(learning_curves_path), str(class_distribution_path), model, scaler)

# Inference engine
def inference_engine(model, scaler, input_data=None, csv_file=None):
    feature_columns = ['time', 'rpm', 'pressure', 'flow_rate', 'temperature',
                       'vibration', 'current', 'noise', 'ambient_temp',
                       'humidity', 'fluid_viscosity', 'external_vibration']

    if csv_file is not None:
        new_data = pd.read_csv(csv_file)
        new_data = new_data[feature_columns]
    elif input_data is not None:
        new_data = pd.DataFrame([input_data], columns=feature_columns)
    else:
        raise ValueError("Either input_data or csv_file must be provided")

    new_data = new_data.fillna(new_data.mean())
    new_data = new_data.apply(pd.to_numeric, errors='coerce')
    new_data = new_data.fillna(new_data.mean())
    new_data_scaled = scaler.transform(new_data)

    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)

    result_df = new_data.copy()
    result_df['predicted_failure_transition'] = predictions
    for i, class_name in enumerate(model.classes_):
        result_df[f'prob_{class_name}'] = probabilities[:, i]

    return result_df

# Predictive maintenance module
def predictive_maintenance(result_df):
    label_counts = result_df['predicted_failure_transition'].value_counts()
    label_proportions = label_counts / len(result_df)

    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.values, y=label_counts.index)
    plt.title('Predicted Failure Transition Distribution')
    plt.xlabel('Count')
    plt.ylabel('Failure Transition')
    label_distribution_path = output_dir / 'inference_label_distribution.png'
    plt.savefig(label_distribution_path)
    plt.close()

    # Estimate data duration
    time_span_hours = result_df['time'].max() - result_df['time'].min()
    duration_text = f"Data Duration: {time_span_hours:.2f} hours"

    # Maintenance timeframe logic
    critical_labels = ['IMPELLER_DAMAGE', 'BEARING', 'CAVITATION']
    moderate_labels = ['OVERLOAD', 'SEAL_LEAK']

    critical_proportion = label_proportions[critical_labels].sum() if any(l in label_proportions for l in critical_labels) else 0
    moderate_proportion = label_proportions[moderate_labels].sum() if any(l in label_proportions for l in moderate_labels) else 0
    safe_proportion = label_proportions.get('SAFE', 0)

    routine_maintenance = 1500
    major_overhaul = 10000

    if critical_proportion > 0.1:
        timeframe = "Immediate (within 1–7 days)"
        urgency = "Urgent"
        urgency_reason = f"High proportion of critical failures ({critical_proportion:.2%})"
    elif moderate_proportion > 0.2:
        timeframe = "Within 100–500 hours (~4–20 days, assuming 24/7 operation)"
        urgency = "High"
        urgency_reason = f"Significant proportion of moderate failures ({moderate_proportion:.2%})"
    elif safe_proportion > 0.8:
        timeframe = f"Next routine maintenance (~{routine_maintenance} hours)"
        urgency = "Low"
        urgency_reason = f"Predominantly SAFE operation ({safe_proportion:.2%})"
    else:
        timeframe = "Within 500–1000 hours (~20–40 days, assuming 24/7 operation)"
        urgency = "Moderate"
        urgency_reason = "Mixed failure types detected"

    # Maintenance recommendations, prioritized by severity
    recommendations = []
    critical_recs = []
    moderate_recs = []
    safe_rec = None

    for label in label_counts.index:
        if label == 'IMPELLER_DAMAGE':
            critical_recs.append("Inspect/replace impeller; check for erosion, imbalance, or material wear.")
        elif label == 'BEARING':
            critical_recs.append("Inspect/replace bearings; verify lubrication and alignment.")
        elif label == 'CAVITATION':
            critical_recs.append("Adjust flow conditions; inspect impeller and casing for pitting.")
        elif label == 'OVERLOAD':
            moderate_recs.append("Check motor, electrical systems, and load conditions.")
        elif label == 'SEAL_LEAK':
            moderate_recs.append("Replace seals; inspect for wear or misalignment.")
        elif label == 'SAFE':
            safe_rec = f"SAFE operation detected ({label_proportions['SAFE']:.2%} of samples); continue regular monitoring."

    # Combine recommendations based on urgency
    if urgency in ["Urgent", "High"]:
        recommendations = critical_recs + moderate_recs
        if safe_rec and safe_proportion > 0:
            recommendations.append(f"Note: {safe_rec}")
    elif urgency == "Moderate":
        recommendations = critical_recs + moderate_recs
        if safe_rec and safe_proportion > 0:
            recommendations.append(f"Note: {safe_rec}")
    else:  # Low urgency
        recommendations = critical_recs + moderate_recs
        if safe_rec:
            recommendations.append(safe_rec)
        else:
            recommendations.append("Continue regular monitoring.")

    # Format maintenance analysis
    maintenance_text = f"""
Predictive Maintenance Analysis:
Label Distribution:
{label_counts.to_string()}
Label Proportions:
{label_proportions.to_string()}
Urgency: {urgency}
Reason: {urgency_reason}
Recommended Maintenance Timeframe: {timeframe}
Maintenance Recommendations:
""" + "\n".join(f"- {rec}" for rec in recommendations if rec)

    # Timeline plot
    plt.figure(figsize=(10, 2))
    plt.axvline(0, color='green', label='Current Time')
    if 'Immediate' in timeframe:
        plt.axvspan(0, 168, alpha=0.3, color='red', label='Maintenance Window')
    elif '100–500' in timeframe:
        plt.axvspan(100, 500, alpha=0.3, color='orange', label='Maintenance Window')
    elif '500–1000' in timeframe:
        plt.axvspan(500, 1000, alpha=0.3, color='yellow', label='Maintenance Window')
    else:
        plt.axvspan(routine_maintenance - 100, routine_maintenance + 100, alpha=0.3, color='green', label='Maintenance Window')
    plt.title('Estimated Maintenance Timeline')
    plt.xlabel('Operating Hours from Now')
    plt.yticks([])
    plt.legend()
    timeline_path = output_dir / 'maintenance_timeline.png'
    plt.savefig(timeline_path)
    plt.close()

    return maintenance_text, str(label_distribution_path), str(timeline_path), duration_text, result_df

# Gradio interface function
def run_predictive_maintenance(training_csv, inference_csv):
    # Train the model
    (clf_report, confusion_matrix_path, feature_importance_path,
     learning_curves_path, class_distribution_path, model, scaler) = train_model(training_csv)

    if model is None:
        return (clf_report, None, None, None, None, None, None, None, None, None)

    # Run inference and predictive maintenance
    result_df = inference_engine(model, scaler, csv_file=inference_csv)
    maintenance_text, label_distribution_path, timeline_path, duration_text, result_df = predictive_maintenance(result_df)

    # Save inference results
    inference_results_path = output_dir / 'inference_results.csv'
    result_df.to_csv(inference_results_path, index=False)

    return (clf_report, confusion_matrix_path, feature_importance_path,
            learning_curves_path, class_distribution_path, maintenance_text,
            label_distribution_path, timeline_path, duration_text, result_df)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Centrifuge Predictive Maintenance System")
    gr.Markdown("Upload the training dataset and an inference CSV to analyze centrifuge health and predict maintenance needs.")

    with gr.Row():
        with gr.Column():
            training_csv = gr.File(label="Upload Training CSV (merged_dataset.csv)")
            inference_csv = gr.File(label="Upload Inference CSV")
            submit_button = gr.Button("Run Analysis")

        with gr.Column():
            clf_report = gr.Textbox(label="Classification Report")
            duration_text = gr.Textbox(label="Data Duration")
            maintenance_text = gr.Textbox(label="Predictive Maintenance Analysis")

    with gr.Row():
        confusion_matrix_img = gr.Image(label="Confusion Matrix")
        feature_importance = gr.Image(label="Feature Importance")

    with gr.Row():
        learning_curves = gr.Image(label="Learning Curves")
        class_distribution = gr.Image(label="Class Distribution")

    with gr.Row():
        label_distribution = gr.Image(label="Inference Label Distribution")
        timeline = gr.Image(label="Maintenance Timeline")

    inference_results = gr.Dataframe(label="Inference Results")

    submit_button.click(
        fn=run_predictive_maintenance,
        inputs=[training_csv, inference_csv],
        outputs=[clf_report, confusion_matrix_img, feature_importance,
                 learning_curves, class_distribution, maintenance_text,
                 label_distribution, timeline, duration_text, inference_results]
    )

# Launch Gradio interface
demo.launch(share=True, debug=True)
