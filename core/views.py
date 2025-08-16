from django.shortcuts import render, redirect
from .forms import HouseForms, RegisterForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import HouseForms, RegisterForm

import xgboost as xgb
import joblib
import numpy as np


# Load model and scaler
model = xgb.XGBRegressor()
model.load_model("ml_model/model.json")
scaler = joblib.load("ml_model/scaler.plk")

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    form = LoginForm(request.POST or None)
    message = ''

    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                message = 'Invalid username or password.'

    return render(request, 'core/login.html', {'form': form, 'message': message})

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'core/register.html', {'form': form})


@login_required
def home_view(request):
    if request.method == 'POST':
        form = HouseForms(request.POST)
        if form.is_valid():
            data = [
                form.cleaned_data['MedInc'],
                form.cleaned_data['HouseAge'],
                form.cleaned_data['AveRooms'],
                form.cleaned_data['AveBedrms'],
                form.cleaned_data['Population'],
                form.cleaned_data['AveOccup']
            ]
            scaled_data = scaler.transform([data])
            prediction = model.predict(scaled_data)[0]
            prediction = prediction * 100000
            return render(request, 'core/results.html', {'result': round(prediction, 2)})
    else:
        form = HouseForms()
    return render(request, 'core/home.html', {'form': form})

@login_required
def results_view(request):
    return redirect('home')

def logout_view(request):
    logout(request)
    return redirect('login')
