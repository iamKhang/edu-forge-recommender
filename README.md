# Edu Forge Recommender

A Django REST Framework application with TensorFlow integration for course recommendations.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start the development server:
```bash
python manage.py runserver
```

## API Endpoints

- `/api/courses/` - List and create courses
- `/api/users/` - List and create users
- `/api/interactions/` - List and create user-course interactions
- `/api/interactions/get_recommendations/?user_id=<id>` - Get course recommendations for a user

## Models

- Course: Represents educational courses
- User: Represents system users
- UserCourseInteraction: Tracks user interactions with courses

## Technologies Used

- Django
- Django REST Framework
- TensorFlow
- NumPy
- Pandas