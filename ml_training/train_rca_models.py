"""
Root Cause Analysis Model Training Script
==========================================
Trains Decision Tree, Random Forest, and Neural Network models for IT incident RCA
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==================== DATA GENERATION ====================
def generate_incident_data(n_samples=1000):
    """
    Generate realistic IT incident data for training
    
    Returns:
        DataFrame with incident features and root causes
    """
    print(f"Generating {n_samples} incident records...")
    
    # Define possible values
    categories = ['Network', 'Database', 'Application', 'Hardware', 'Security']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    systems = ['Web Server', 'Database Server', 'API Gateway', 'Load Balancer', 'Storage']
    root_causes = [
        'Configuration Error',
        'Resource Exhaustion',
        'Network Congestion',
        'Database Deadlock',
        'Memory Leak',
        'Disk Full',
        'Authentication Failure',
        'Bug in Code'
    ]
    
    # Generate synthetic data with realistic patterns
    data = []
    for _ in range(n_samples):
        category = np.random.choice(categories)
        priority = np.random.choice(priorities)
        system = np.random.choice(systems)
        error_code = np.random.randint(100, 999)
        
        # Create correlations between features and root causes
        if category == 'Network':
            root_cause = np.random.choice(['Network Congestion', 'Configuration Error', 'Resource Exhaustion'], 
                                         p=[0.5, 0.3, 0.2])
        elif category == 'Database':
            root_cause = np.random.choice(['Database Deadlock', 'Resource Exhaustion', 'Disk Full'], 
                                         p=[0.4, 0.4, 0.2])
        elif category == 'Application':
            root_cause = np.random.choice(['Bug in Code', 'Memory Leak', 'Configuration Error'], 
                                         p=[0.5, 0.3, 0.2])
        elif category == 'Hardware':
            root_cause = np.random.choice(['Disk Full', 'Resource Exhaustion', 'Configuration Error'], 
                                         p=[0.4, 0.4, 0.2])
        else:  # Security
            root_cause = np.random.choice(['Authentication Failure', 'Configuration Error', 'Bug in Code'], 
                                         p=[0.5, 0.3, 0.2])
        
        data.append({
            'category': category,
            'priority': priority,
            'affected_system': system,
            'error_code': error_code,
            'root_cause': root_cause
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} incidents")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  Root Causes: {df['root_cause'].nunique()}")
    return df

# ==================== DATA PREPROCESSING ====================
def preprocess_data(df):
    """
    Encode categorical features and prepare for training
    
    Returns:
        X, y, encoders dictionary
    """
    print("\nPreprocessing data...")
    
    # Initialize label encoders
    encoders = {
        'category': LabelEncoder(),
        'priority': LabelEncoder(),
        'system': LabelEncoder(),
        'root_cause': LabelEncoder()
    }
    
    # Encode categorical features
    df['category_encoded'] = encoders['category'].fit_transform(df['category'])
    df['priority_encoded'] = encoders['priority'].fit_transform(df['priority'])
    df['system_encoded'] = encoders['system'].fit_transform(df['affected_system'])
    df['cause_encoded'] = encoders['root_cause'].fit_transform(df['root_cause'])
    
    # Prepare features and target
    X = df[['category_encoded', 'priority_encoded', 'system_encoded', 'error_code']]
    y = df['cause_encoded']
    
    print(f"✓ Feature matrix: {X.shape}")
    print(f"✓ Target classes: {len(np.unique(y))}")
    
    return X, y, encoders

# ==================== MODEL TRAINING ====================
def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate Decision Tree classifier"""
    print("\n" + "="*50)
    print("TRAINING DECISION TREE")
    print("="*50)
    
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, accuracy

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier"""
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_names = ['Category', 'Priority', 'System', 'Error Code']
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")
    
    return model, accuracy

def train_neural_network(X_train, y_train, X_test, y_test):
    """Train and evaluate Neural Network classifier"""
    print("\n" + "="*50)
    print("TRAINING NEURAL NETWORK")
    print("="*50)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training iterations: {model.n_iter_}")
    
    return model, accuracy

# ==================== VISUALIZATION ====================
def plot_confusion_matrix(y_test, y_pred, class_names, model_name, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved confusion matrix to {save_path}")

def plot_feature_importance(model, feature_names, save_path):
    """Plot feature importance for tree-based models"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved feature importance to {save_path}")

def plot_model_comparison(results, save_path):
    """Compare model accuracies"""
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved model comparison to {save_path}")

# ==================== MAIN TRAINING PIPELINE ====================
def main():
    print("="*70)
    print("RCA MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Generate data
    df = generate_incident_data(n_samples=1000)
    
    # Step 2: Preprocess
    X, y, encoders = preprocess_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train models
    results = {}
    
    dt_model, dt_acc = train_decision_tree(X_train, y_train, X_test, y_test)
    results['Decision Tree'] = {'model': dt_model, 'accuracy': dt_acc}
    
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = {'model': rf_model, 'accuracy': rf_acc}
    
    nn_model, nn_acc = train_neural_network(X_train, y_train, X_test, y_test)
    results['Neural Network'] = {'model': nn_model, 'accuracy': nn_acc}
    
    # Step 5: Generate visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    class_names = encoders['root_cause'].classes_
    feature_names = ['Category', 'Priority', 'System', 'Error Code']
    
    # Confusion matrices
    for model_name, data in results.items():
        y_pred = data['model'].predict(X_test)
        plot_confusion_matrix(y_test, y_pred, class_names, model_name, 
                            f'/app/ml_training/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    # Feature importance for tree models
    plot_feature_importance(rf_model, feature_names, 
                          '/app/ml_training/feature_importance.png')
    
    # Model comparison
    plot_model_comparison(results, '/app/ml_training/model_comparison.png')
    
    # Step 6: Save models and encoders
    print("\n" + "="*50)
    print("SAVING MODELS AND ENCODERS")
    print("="*50)
    
    output_dir = '/app/backend/ml_models/'
    
    pickle.dump(dt_model, open(output_dir + 'decision_tree.pkl', 'wb'))
    print(f"  ✓ Saved Decision Tree model")
    
    pickle.dump(rf_model, open(output_dir + 'random_forest.pkl', 'wb'))
    print(f"  ✓ Saved Random Forest model")
    
    pickle.dump(nn_model, open(output_dir + 'neural_network.pkl', 'wb'))
    print(f"  ✓ Saved Neural Network model")
    
    for name, encoder in encoders.items():
        filename = f'le_{name}.pkl'
        pickle.dump(encoder, open(output_dir + filename, 'wb'))
        print(f"  ✓ Saved {name} encoder")
    
    # Step 7: Generate classification reports
    print("\n" + "="*50)
    print("CLASSIFICATION REPORTS")
    print("="*50)
    
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        y_pred = data['model'].predict(X_test)
        print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nModel Accuracies:")
    for model_name, data in results.items():
        print(f"  {model_name}: {data['accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    
    print("\nOutput Files:")
    print(f"  - Models saved to: {output_dir}")
    print(f"  - Visualizations saved to: /app/ml_training/")

if __name__ == "__main__":
    main()
