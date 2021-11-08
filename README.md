<img src='images/King County.jpeg'>

**King County, WA, U.S.A.**  

# Housing Guidance for King County, WA, U.S.A

# Overview

**Client:** New WA state home buyers needing consultation on WA real estate market and expectations (price, size, location) 

**Data, Methodology, and Analysis:** Using King County (WA, U.S.A.) housing data from 2014-2015, linear regression models were built to understand the relationship between price and all other variables in the data (square feet, bedrooms, bathrooms, zipcodes, year built, year renovated, waterfront views)

**Results & Recommendations:** After analyzing data and building linear regression models assessing relationships between price and square feet; price and bedrooms; and price to zip code, the expectations for price range were modeled depending on square feet of living space, grade, condition, and renovation status 


# RESULTS - FINAL MODEL

** Final Model includes: 

1. Bedrooms 
2. Bathrooms 
3. Square Feet Living
4. Floors
5. Waterfront 
6. Condition
7. Grade
8. Renovation Status
9. Basement Present
10. All Zipcodes

Excluding zipcodes, **Square Feet Living, Grade, and Condition are the strongest determinants of price** 

<img src='images/Final Model.png'>

<img src='images/Square Feet Living.png'>

<img src='images/Building Grade.png'>

<img src='images/Condition.png'>


## Check for Linearity and Residual Normality using Q-Q Plot 
- There are some tails but overall residuals appear normal 

<img src='images/Residual Normality.png'>
    

## Check for Homoskedasticity

- Durbin-Watson: range of 1.5 to 2.5 is relatively normal
- Model's Durbin-Watson is 2.024

## Check for Over-fitting
- Check expected vs. predicted errors of Train Test sets 
- Train and Test data are within range of each other 
- The average expected error (mean absolute error) of the Train data is \\$47,496 while the average expected error of the Test data is \\$48,684


# Evaluation and Conclusions

After building models to evaluate the relationship between price and several factors, we can offer guidance to new home buyers in WA State about the expectation of price relative to square feet of living, waterfront views, condition, and grade. 

**** Important note: the results are best suited for home buyers seeking homes with a maximum of 6 bedrooms, 4000 square feet, and a budget ranging from \\$175,000 to \\$650,000

**Conclusions** 
- The **average price for a home in King County, WA is approximately \\$400,358**
-  **The most important factors in our model, besides zipcode, are: Square Feet Living, Grade, and Condition**
- **Every additional square feet of space costs approximately \\$94. Note: other models showed this cost could be up to $200 per square feet in the densest zipcodes** 
- The grade of a home (1-13) is a strong determinant of price. **Every grade increase costs approximately \\$31,571** 
- The condition (1-5) is also a strong determinant; **Every condition level increase costs approximately \\$15,007** 
- **If the home has been renovated, the price is expected to be approximately \\$9812 more**


# Future Work 

**Future work:** 
* Refine existing models and expand dataset for different types of home buyers 
* Explore relationship of price to zip code 
* Build models for Suburbs (Medina, WA) vs. City (Seattle, WA)
* Build more comprehensive models considering other factors such as location, renovations, waterfront view 


# Repository Structure

- images 
- README.me
- Vi_Bui_Phase2_Final_Project_Presentation.pdf
- Vi_Bui_Phase2_Project_FinalFinal - Jupyter Notebook.pdf
- Vi_Bui_Phase2_Project_FinalFinal.ipynb
