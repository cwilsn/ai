<?php

use Aws\MockHandler;
use Aws\Result;
use Illuminate\JsonSchema\JsonSchemaTypeFactory;
use Laravel\Ai\Attributes\CacheInstructions;
use Laravel\Ai\Exceptions\NoSuchToolException;
use Laravel\Ai\Gateway\TextGenerationLoop;
use Laravel\Ai\Gateway\TextGenerationOptions;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\Responses\StructuredTextResponse;
use Laravel\Ai\Responses\TextResponse;
use Laravel\Ai\Streaming\Events\StreamEnd;
use Laravel\Ai\Streaming\Events\ToolResult as ToolResultEvent;
use Tests\Fixtures\Tools\FixedNumberGenerator;

function bedrockToolCallResponse(string $toolUseId): array
{
    return [
        'output' => ['message' => ['content' => [
            ['toolUse' => ['toolUseId' => $toolUseId, 'name' => 'FixedNumberGenerator', 'input' => []]],
        ]]],
        'usage' => ['inputTokens' => 7, 'outputTokens' => 3],
        'stopReason' => 'tool_use',
    ];
}

function bedrockTextResponse(string $text): array
{
    return [
        'output' => ['message' => ['content' => [['text' => $text]]]],
        'usage' => ['inputTokens' => 7, 'outputTokens' => 5],
        'stopReason' => 'end_turn',
    ];
}

describe('tool call loop', function (): void {
    test('multi step tool loop returns accumulated response shape', function (): void {
        $mock = new MockHandler([
            new Result(bedrockToolCallResponse('t1')),
            new Result(bedrockToolCallResponse('t2')),
            new Result(bedrockTextResponse('Done')),
        ]);

        $gateway = $this->gatewayWithClient($this->bedrockClient($mock));

        $response = (new TextGenerationLoop($gateway))->generate(
            $this->bedrockProvider(),
            'anthropic.claude-opus-4-7-v1:0',
            null,
            messages: [new UserMessage('Generate numbers')],
            tools: [new FixedNumberGenerator],
            options: new TextGenerationOptions(maxSteps: 5),
        );

        expect((string) $response)->toBe('Done')
            ->and($response->messages)->toHaveCount(5)
            ->and($response->steps)->toHaveCount(3)
            ->and($response->toolCalls)->toHaveCount(2)
            ->and($response->toolResults)->toHaveCount(2)
            ->and($response->usage->inputTokens)->toBe(21)
            ->and($response->usage->outputTokens)->toBe(11);

        $parameters = $mock->getLastCommand()->toArray();

        expect($parameters['toolConfig'])->toHaveKeys(['tools'])
            ->not->toHaveKey('toolChoice')
            ->and($parameters)->not->toHaveKey('system');
    });

    test('max steps limits tool call depth', function (): void {
        $client = $this->fakeBedrockConverseSequence([
            bedrockToolCallResponse('t1'),
            bedrockToolCallResponse('t2'),
            bedrockToolCallResponse('t3'),
            bedrockTextResponse('Done'),
        ]);

        $gateway = $this->gatewayWithClient($client);

        $response = (new TextGenerationLoop($gateway))->generate(
            $this->bedrockProvider(),
            'anthropic.claude-opus-4-7-v1:0',
            null,
            messages: [new UserMessage('Generate numbers')],
            tools: [new FixedNumberGenerator],
            options: new TextGenerationOptions(maxSteps: 2),
        );

        expect($response->steps)->toHaveCount(2);
    });

    test('unknown tool call throws NoSuchToolException', function (): void {
        $client = $this->fakeBedrockConverse([
            'output' => ['message' => ['content' => [
                ['toolUse' => ['toolUseId' => 't1', 'name' => 'NonExistentTool', 'input' => []]],
            ]]],
            'usage' => ['inputTokens' => 7, 'outputTokens' => 3],
            'stopReason' => 'tool_use',
        ]);

        $gateway = $this->gatewayWithClient($client);

        expect(fn (): TextResponse => (new TextGenerationLoop($gateway))->generate(
            $this->bedrockProvider(),
            'anthropic.claude-opus-4-7-v1:0',
            null,
            tools: [new FixedNumberGenerator],
        ))->toThrow(NoSuchToolException::class);
    });

    test('structured output uses auto tool choice and is parsed from the synthetic tool call', function (?string $instructions, ?array $providerSystem, bool $cacheInstructions): void {
        $mock = new MockHandler([new Result([
            'output' => ['message' => ['content' => [
                ['toolUse' => ['toolUseId' => 's1', 'name' => 'structured_output', 'input' => ['symbol' => 'Fe']]],
            ]]],
            'usage' => ['inputTokens' => 8, 'outputTokens' => 4],
            'stopReason' => 'tool_use',
        ])]);

        $gateway = $this->gatewayWithClient($this->bedrockClient($mock));

        $response = (new TextGenerationLoop($gateway))->generate(
            $this->bedrockProvider(),
            'bedrock-model',
            $instructions,
            schema: ['symbol' => (new JsonSchemaTypeFactory)->string()],
            options: new TextGenerationOptions(
                cacheInstructions: $cacheInstructions ? new CacheInstructions : null,
                providerOptions: $providerSystem !== null ? ['system' => $providerSystem] : null,
            ),
        );

        expect($response)->toBeInstanceOf(StructuredTextResponse::class)
            ->and($response->structured)->toMatchArray(['symbol' => 'Fe'])
            ->and($response->steps)->toHaveCount(1)
            ->and($response->usage->inputTokens)->toBe(8)
            ->and($response->usage->outputTokens)->toBe(4);

        $parameters = $mock->getLastCommand()->toArray();
        $originalSystem = $providerSystem ?? ($instructions ? [['text' => $instructions]] : []);
        $system = $parameters['system'][count($originalSystem)]['text'] ?? '';

        expect($parameters['toolConfig']['toolChoice'])->toBe(['auto' => []])
            ->and($parameters['toolConfig']['tools'])->toHaveCount(1)
            ->and($parameters['toolConfig']['tools'][0]['toolSpec']['name'])->toBe('structured_output')
            ->and(array_slice($parameters['system'], 0, count($originalSystem)))->toBe($originalSystem)
            ->and($parameters['system'])->toHaveCount(count($originalSystem) + 1 + (int) $cacheInstructions)
            ->and(substr_count(implode("\n", array_column($parameters['system'], 'text')), 'structured_output'))->toBe(1)
            ->and($system)->toContain('When you are ready to provide your final answer', 'must return it by calling the structured_output tool', 'Do not return the final answer as plain text');

        if ($cacheInstructions) {
            expect(end($parameters['system']))->toBe(['cachePoint' => ['type' => 'default']]);
        }
    })->with([
        'no instructions' => [null, null, false],
        'empty instructions' => ['', null, false],
        'agent instructions' => ['You are a helpful assistant.', null, false],
        'provider system' => ['Overridden agent instructions.', [['text' => 'Provider instructions.'], ['text' => 'Additional instructions.']], false],
        'empty provider system' => ['Overridden agent instructions.', [], false],
        'cached agent instructions' => ['You are a helpful assistant.', null, true],
        'cached provider system' => ['Overridden agent instructions.', [['text' => 'Provider instructions.']], true],
    ]);

    test('structured output keeps normal tools available with auto selection through the final step', function (): void {
        $parameters = [];
        $results = [
            bedrockToolCallResponse('t1'),
            bedrockToolCallResponse('t2'),
            [
                'output' => ['message' => ['content' => [
                    ['toolUse' => ['toolUseId' => 's1', 'name' => 'structured_output', 'input' => ['number' => 42]]],
                ]]],
                'stopReason' => 'tool_use',
            ],
        ];

        $mock = new MockHandler(array_map(function (array $result) use (&$parameters) {
            return function ($command) use ($result, &$parameters): Result {
                $parameters[] = $command->toArray();

                return new Result($result);
            };
        }, $results));

        $response = (new TextGenerationLoop($this->gatewayWithClient($this->bedrockClient($mock))))->generate(
            $this->bedrockProvider(),
            'bedrock-model',
            'Generate numbers before answering.',
            messages: [new UserMessage('Generate numbers')],
            tools: [new FixedNumberGenerator],
            schema: ['number' => (new JsonSchemaTypeFactory)->integer()],
            options: new TextGenerationOptions(maxSteps: 3),
        );

        expect($response)->toBeInstanceOf(StructuredTextResponse::class)
            ->and($response->structured)->toBe(['number' => 42])
            ->and($response->toolResults)->toHaveCount(2)
            ->and($parameters)->toHaveCount(3);

        foreach ($parameters as $request) {
            expect($request['toolConfig']['toolChoice'])->toBe(['auto' => []])
                ->and(array_column(array_column($request['toolConfig']['tools'], 'toolSpec'), 'name'))->toBe(['structured_output', 'FixedNumberGenerator'])
                ->and($request['system'])->toBe($parameters[0]['system'])
                ->and(substr_count(implode("\n", array_column($request['system'], 'text')), 'structured_output'))->toBe(1);
        }
    });

    test('streaming tool loop emits a single stream end with accumulated usage', function (): void {
        $client = $this->fakeBedrockStreamSequence([
            [
                $this->contentBlockStart(0, ['toolUse' => ['toolUseId' => 't1', 'name' => 'FixedNumberGenerator']]),
                $this->contentBlockDelta(0, ['toolUse' => ['input' => '{}']]),
                $this->contentBlockStop(0),
                $this->messageStop('tool_use'),
                ['metadata' => ['usage' => ['inputTokens' => 5, 'outputTokens' => 2]]],
            ],
            [
                $this->contentBlockStart(0),
                $this->contentBlockDelta(0, ['text' => 'Done']),
                $this->contentBlockStop(0),
                $this->messageStop('end_turn'),
                ['metadata' => ['usage' => ['inputTokens' => 5, 'outputTokens' => 2]]],
            ],
        ]);

        $gateway = $this->gatewayWithClient($client);

        $events = iterator_to_array(
            (new TextGenerationLoop($gateway))->stream(
                'inv-1',
                $this->bedrockProvider(),
                'anthropic.claude-opus-4-7-v1:0',
                null,
                tools: [new FixedNumberGenerator],
            ),
            preserve_keys: false,
        );

        $streamEnds = array_values(array_filter($events, fn ($e): bool => $e instanceof StreamEnd));
        $toolResults = array_values(array_filter($events, fn ($e): bool => $e instanceof ToolResultEvent));

        expect($streamEnds)->toHaveCount(1)
            ->and($streamEnds[0]->reason)->toBe('stop')
            ->and($streamEnds[0]->usage->inputTokens)->toBe(10)
            ->and($streamEnds[0]->usage->outputTokens)->toBe(4)
            ->and($toolResults)->toHaveCount(1)
            ->and($events[count($events) - 1])->toBeInstanceOf(StreamEnd::class);
    });
});
