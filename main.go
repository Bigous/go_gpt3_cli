package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
)

const apiURL = "https://api.openai.com/v1/completions"

type CompletionRequest struct {
	Prompt      string  `json:"prompt"`      // O texto de entrada para o qual deseja-se completar
	Model       string  `json:"model"`       // O nome do modelo a ser usado para completar o texto
	MaxTokens   int     `json:"max_tokens"`  // O número máximo de tokens (palavras e pontuações) a serem retornados na resposta
	Temperature float64 `json:"temperature"` // O nível de temperatura a ser usado para controlar a aleatoriedade do modelo
}

type CompletionResponse struct {
	Choices []Choice `json:"choices"` // As opções de compleção geradas pelo modelo
}

type Choice struct {
	Text string `json:"text"` // O texto gerado pelo modelo
}

func generateText(prompt string, model string) (string, error) {
	if model == "" {
		model = "text-davinci-003"
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", errors.New("OPENAI_API_KEY environment variable not set")
	}

	requestBody, err := json.Marshal(CompletionRequest{
		Prompt:      prompt,
		Model:       model,
		Temperature: 0.8,
		MaxTokens:   2000,
	})

	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("received non-200 status code: %d", resp.StatusCode)
	}

	// Deserialize a resposta
	var completionResponse CompletionResponse
	err = json.NewDecoder(resp.Body).Decode(&completionResponse)
	if err != nil {
		return "", err
	}

	return completionResponse.Choices[0].Text, nil
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter prompt: ")
	prompt, _ := reader.ReadString('\n')
	prompt = strings.TrimSpace(prompt)

	result, err := generateText(prompt, "")
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(result)
}
