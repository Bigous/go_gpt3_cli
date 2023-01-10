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
	Prompt            string   `json:"prompt"`               // O texto de entrada para o qual deseja-se completar
	Model             string   `json:"model"`                // O nome do modelo a ser usado para completar o texto
	MaxTokens         int      `json:"max_tokens"`           // O número máximo de tokens (palavras e pontuações) a serem retornados na resposta
	Temperature       float64  `json:"temperature"`          // O nível de temperatura a ser usado para controlar a aleatoriedade do modelo
	TopP              float64  `json:"top_p"`                // O limite superior da probabilidade a ser usado para filtrar as opções de compleção
	FrequencyPenalty  float64  `json:"frequency_penalty"`    // A penalidade de frequência a ser aplicada às palavras mais comuns
	PresencePenalty   float64  `json:"presence_penalty"`     // A penalidade de presença a ser aplicada às palavras presentes no prompt de entrada
	BestOf            int      `json:"best_of"`              // O número de opções de compleção a serem retornadas (1-5)
	Stop              string   `json:"stop"`                 // A sequência de tokens que fará com que a geração de texto seja interrompida
	Stream            bool     `json:"stream"`               // Indica se a resposta deve ser retornada como um stream de dados
	MaxTokensPerBatch int      `json:"max_tokens_per_batch"` // O número máximo de tokens a serem retornados em cada batch em uma resposta em stream
	N                 int      `json:"n"`                    // O número de opções de compleção a serem retornadas (1-5)
	Streaming         bool     `json:"streaming"`            // Indica se a resposta deve ser retornada como um stream de dados
	StopSequence      []string `json:"stop_sequence"`        // A sequência de tokens que fará com que a geração de texto seja interrompida
	BatchSize         int      `json:"batch_size"`           // O número de opções de compleção a serem retornadas em cada batch em uma resposta em stream
	MaxTries          int      `json:"max_tries"`            // O número máximo de tentativas a serem realizadas para obter uma resposta válida em uma solicitação em stream
	Diversity         float64  `json:"diversity"`            // O número máximo de tentativas a serem realizadas para obter uma resposta válida em uma solicitação em stream
}

type CompletionResponse struct {
	ID                string   `json:"id"`                   // O ID da solicitação de compleção
	Model             string   `json:"model"`                // O nome do modelo usado para completar o texto
	Prompt            string   `json:"prompt"`               // O prompt de entrada fornecido na solicitação de compleção
	Choices           []Choice `json:"choices"`              // As opções de compleção geradas pelo modelo
	Stream            bool     `json:"stream"`               // Indica se a resposta foi retornada como um stream de dados
	Streaming         bool     `json:"streaming"`            // Indica se a resposta foi retornada como um stream de dados
	Truncated         bool     `json:"truncated"`            // Indica se a resposta foi truncada devido ao tamanho máximo permitido
	Warning           string   `json:"warning"`              // Uma mensagem de aviso, se houver
	Errors            []string `json:"errors"`               // Uma lista de erros, se houver
	HasMore           bool     `json:"has_more"`             // Indica se há mais opções de compleção disponíveis em uma resposta em stream
	MaxTokensPerBatch int      `json:"max_tokens_per_batch"` // O número máximo de tokens a serem retornados em cada batch em uma resposta em stream
	BatchSize         int      `json:"batch_size"`           // O número de opções de compleção a serem retornadas em cada batch em uma resposta em stream
	MaxTries          int      `json:"max_tries"`            // O número máximo de tentativas a serem realizadas para obter uma resposta válida em uma solicitação em stream
}

type Choice struct {
	Text             string  `json:"text"`              // O texto gerado pelo modelo
	Index            int     `json:"index"`             // O índice da opção de compleção (começando em 0)
	Logprobs         []int   `json:"logprobs"`          // As probabilidades logarítmicas das palavras geradas pelo modelo
	FinishReason     string  `json:"finish_reason"`     // A razão pela qual a geração de texto foi interrompida
	PresencePenalty  float64 `json:"presence_penalty"`  // A penalidade de presença aplicada às palavras presentes no prompt de entrada
	FrequencyPenalty float64 `json:"frequency_penalty"` // A penalidade de frequência aplicada às palavras mais comuns
	Reranked         bool    `json:"reranked"`          // Indica se a opção de compleção foi reclassificada
	Truncated        bool    `json:"truncated"`         // Indica se a opção de compleção foi truncada devido ao tamanho máximo permitido
	Diversity        float64 `json:"diversity"`         // O nível de diversidade da opção de compleção
	Temperature      float64 `json:"temperature"`       // O nível de temperatura usado para controlar a aleatoriedade do modelo
	TopP             float64 `json:"top_p"`             // O limite superior da probabilidade usado para filtrar as opções de compleção
	StopSequence     []int   `json:"stop_sequence"`     // A sequência de tokens que fará com que a geração de texto seja interrompida
	Tokens           []int   `json:"tokens"`            // A lista de tokens gerados pelo modelo
	Weight           float64 `json:"weight"`            // O peso da opção de compleção
	Probability      float64 `json:"probability"`       // A probabilidade da opção de compleção
	Rank             int     `json:"rank"`              // A classificação da opção de compleção
	DiversityRank    int     `json:"diversity_rank"`    // A classificação da diversidade da opção de compleção
	DiversityScore   float64 `json:"diversity_score"`   // A pontuação de diversidade da opção de compleção
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
