// UI-TARS-SDK.ts
/// <reference types="node" />

import { GUIAgent } from '@ui-tars/sdk';
import { NutJSOperator } from '@ui-tars/operator-nut-js';
import { UITarsModelVersion } from '@ui-tars/shared/constants'; // will be from '@ui-tars/sdk';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';


(async () => {
  const { params } = yargs(hideBin(process.argv)).option('params', {
    type: 'string',
    demandOption: true,
  }).parseSync();

  const paramJSON = JSON.parse(params);
  const config = paramJSON.config
  const life_time = parseInt(paramJSON.life_time) ?? 10

  const abortController = new AbortController();
  process.on('SIGINT', () => abortController.abort());

  const agent = new GUIAgent({
    model: {
      baseURL: config.baseURL,
      apiKey: config.apiKey,
      model: config.model,
    },
    operator: new NutJSOperator(),
    /////* Followings are optional parameters */////
    // systemPrompt?: string;
    signal: abortController.signal,
    onData: ({ data }) => console.log(data),
    onError: ({ data, error }) => {
      const reachedMax = error?.status === -100004;
      if (reachedMax) {
        console.log('[UITarsModel] reached max iterations: Life-time=' + life_time);
        // return successfully exit signal: 100 (Any number 0~255 without OS conflict can be used)
        process.exit(100);
      }
      console.error(error, data);
      process.exit(1);
    },
    // logger?: Logger;
    // retry?: {
    //   model?: RetryConfig;
    //   /** TODO: whether need to provider retry config in SDK?, should be provided with operator? */
    //   screenshot?: RetryConfig;
    //   execute?: RetryConfig;
    // };
    maxLoopCount: life_time,
    loopIntervalInMs: 1000,  // Time interval between two loop iterations (in milliseconds), @default 0
    uiTarsVersion: UITarsModelVersion.V1_5, // hardcoded. Task: import converter like getModelVersion(not exported)
  });

  await agent.run(paramJSON.task);
})().catch(err => {
  console.error('[UI-TARS] fatal', err);
  process.exit(1);
});