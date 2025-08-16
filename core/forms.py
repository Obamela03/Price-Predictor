from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

class HouseForms(forms.Form):
    MedInc = forms.FloatField(min_value=1, max_value=15, label='Median Income [1 - 15]')
    HouseAge = forms.FloatField(min_value=1, max_value=52, label='House Age [1 - 50]')
    AveRooms = forms.FloatField(min_value=1, max_value=150, label='Average Rooms [1 - 150]')
    AveBedrms = forms.FloatField(min_value=0, max_value=34, label='Average Bedrooms [1 - 35]')
    Population = forms.FloatField(min_value=0, label='Population')
    AveOccup = forms.FloatField(min_value=0, label='Average Occupancy')


    def clean_RM(self):
        rm = self.cleaned_data['RM']
        if rm < 1.0:
            raise forms.ValidationError("Average rooms are too low.")
        return rm

    def clean_PTRATIO(self):
        ratio = self.cleaned_data['PTRATIO']
        if ratio > 30:
            raise forms.ValidationError("PTRATIO too high, check again.")
        return ratio
    
class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class LoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        label='Username',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username'
        })
    )
    
    password = forms.CharField(
        max_length=128,
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
    )
