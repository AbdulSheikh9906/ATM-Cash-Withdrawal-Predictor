library(shiny)
library(shinythemes)
library(ggplot2)
library(readr)
library(lubridate)
library(dplyr)
library(Metrics)
library(zoo)
library(randomForest)
library(caret)

# Load the saved model and preprocessed data
load("D:/DS_Lab/CP/ATM-withdrawal-prediction-model-main/final_model_improved.rdata")
load("D:/DS_Lab/CP/ATM-withdrawal-prediction-model-main/final_data_used_in_model.rdata")

# UI
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("💰 ATM Cash Withdrawal Predictor"),
  br(),
  
  sidebarLayout(
    sidebarPanel(
      tags$h4("📅 Input Parameters"),
      dateInput("date", "Select a Date:", value = Sys.Date(), format = "yyyy-mm-dd"),
      sliderInput("humidity", "Humidity (%):", min = 0, max = 100, value = 60),
      numericInput("lag_service", "Service Yesterday:", value = 100),
      actionButton("predict", "🔍 Predict", class = "btn-primary"),
      br(), br(),
      tags$small("Note: Compares predictions for Raining and Not Raining scenarios.")
    ),
    
    mainPanel(
      tags$h4("📈 Prediction Results"),
      verbatimTextOutput("prediction"),
      br(),
      downloadButton("downloadData", "📥 Download CSV"),
      br(), br(),
      tags$h4("📊 Model Performance Summary"),
      verbatimTextOutput("model_summary"),
      br(),
      tags$h4("📉 Model Diagnostic Plots"),
      plotOutput("fittedPlot", height = "350px"),
      br(),
      plotOutput("residualPlot", height = "350px"),
      br(),
      plotOutput("residualHist", height = "350px"),
      br(),
      tags$h4("🌟 Feature Importance (Random Forest)"),
      plotOutput("importancePlot", height = "400px")
    )
  )
)

# Server
server <- function(input, output) {
  observeEvent(input$predict, {
    selected_date <- input$date
    weekday <- format(selected_date, "%A")
    day_of_month <- day(selected_date)
    month <- month(selected_date)
    
    weekend <- as.numeric(weekday %in% c("Saturday", "Sunday"))
    end_of_month <- as.numeric(day_of_month %in% c(1, 2, 3, 28, 29, 30, 31))
    synthetic_date <- ymd(paste("2023", month, day_of_month, sep = "-"))
    day_of_week_num <- wday(synthetic_date)
    
    base_data <- data.frame(
      end_of_month = end_of_month,
      weekend = weekend,
      end_of_month_and_weekend = as.numeric(end_of_month & weekend),
      lag_1_service = input$lag_service,
      humidity = input$humidity,
      weekday_Friday = as.numeric(weekday == "Friday"),
      weekday_Saturday = as.numeric(weekday == "Saturday"),
      weekday_Sunday = as.numeric(weekday == "Sunday"),
      weekday_Wednesday = as.numeric(weekday == "Wednesday"),
      weekday_Tuesday = as.numeric(weekday == "Tuesday"),
      weekday_Thursday = as.numeric(weekday == "Thursday"),
      service_day_trend = (day_of_month - 15)^2,
      month = as.numeric(month),
      day_of_week_num = day_of_week_num,
      previous_week_avg_service = input$lag_service
    )
    
    newdata_rain <- base_data
    newdata_rain$weather_main_Rain <- 1
    
    newdata_norain <- base_data
    newdata_norain$weather_main_Rain <- 0
    
    pred_rain <- expm1(predict(model, newdata = newdata_rain))
    pred_norain <- expm1(predict(model, newdata = newdata_norain))
    
    output$prediction <- renderText({
      paste0(
        "🌧️ If Raining: ", format(round(pred_rain, 2), big.mark = ","), "\n",
        "☀️ If Not Raining: ", format(round(pred_norain, 2), big.mark = ",")
      )
    })
    
    output$downloadData <- downloadHandler(
      filename = function() {
        paste0("predictions_", Sys.Date(), ".csv")
      },
      content = function(file) {
        write.csv(data.frame(
          Date = selected_date,
          Raining = round(pred_rain, 2),
          Not_Raining = round(pred_norain, 2)
        ), file, row.names = FALSE)
      }
    )
  })
  
  output$model_summary <- renderText({
    # Predict on training data
    predicted_log <- predict(model, newdata = data)
    predicted <- expm1(predicted_log)  # Back-transform
    actual <- data$service
    
    paste0(
      "✅ RMSE: ", round(rmse(actual, predicted), 2), "\n",
      "✅ MAE: ", round(mae(actual, predicted), 2)
    )
  })
  
  output$fittedPlot <- renderPlot({
    plot_data <- data
    plot_data$fitted_values <- expm1(predict(model, newdata = plot_data))
    
    ggplot(plot_data, aes(x = fitted_values, y = service)) +
      geom_point(color = "#0072B2", alpha = 0.6) +
      geom_smooth(method = "lm", se = FALSE, color = "darkred", linetype = "dashed") +
      labs(x = "Predicted (Fitted) Values", y = "Actual Withdrawals", title = "Fitted vs Actual Withdrawals") +
      theme_minimal()
  })
  
  output$residualPlot <- renderPlot({
    plot_data <- data
    plot_data$fitted_values <- expm1(predict(model, newdata = plot_data))
    plot_data$residuals <- plot_data$service - plot_data$fitted_values
    
    ggplot(plot_data, aes(x = fitted_values, y = residuals)) +
      geom_point(alpha = 0.6, color = "#D55E00") +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
      labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs Fitted Values") +
      theme_minimal()
  })
  
  output$residualHist <- renderPlot({
    plot_data <- data
    plot_data$fitted_values <- expm1(predict(model, newdata = plot_data))
    plot_data$residuals <- plot_data$service - plot_data$fitted_values
    
    ggplot(plot_data, aes(x = residuals)) +
      geom_histogram(fill = "#009E73", color = "white", bins = 30) +
      labs(x = "Residuals", y = "Count", title = "Histogram of Residuals") +
      theme_minimal()
  })
  
  output$importancePlot <- renderPlot({
    varImp_df <- varImp(model, scale = TRUE)$importance
    varImp_df$Feature <- rownames(varImp_df)
    
    ggplot(varImp_df, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_col(fill = "#56B4E9") +
      coord_flip() +
      labs(x = "Feature", y = "Importance Score", title = "Feature Importance from Random Forest") +
      theme_minimal()
  })
}

# Run the app
shinyApp(ui = ui, server = server)