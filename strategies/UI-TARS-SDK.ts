// UI-TARS-SDK.ts
/// <reference types="node" />

import { GUIAgent } from '@ui-tars/sdk';
import { NutJSOperator } from '@ui-tars/operator-nut-js';
import { UITarsModelVersion } from '@ui-tars/shared/constants';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';

const { params } = yargs(hideBin(process.argv))
  .option('params', { type: 'string', demandOption: true })
  .parseSync();

const paramJSON  = JSON.parse(params as string);
const config     = paramJSON.config;
const life_time  = Number(paramJSON.life_time) || 10;

const abortController = new AbortController();
process.on('SIGINT', () => abortController.abort());

const agent = new GUIAgent({
  model: {
    baseURL : config.baseURL,
    apiKey  : config.apiKey,
    model   : config.model,
  },
  operator: new NutJSOperator(),
  signal: abortController.signal,
  onData: ({ data }) => console.log(data),

  onError: ({ data, error }) => {
    const reachedMax = error?.status === -100004;
    if (reachedMax) {
      console.log(`[UITarsModel] reached max iterations: Life-time=${life_time}`);
      process.exit(100);
    }

    /* invoke failure（413: message length over the limit of Hugging Face endpoint）*/
    if (error?.status === -100001) {
      console.log('[UITarsModel] model-invoke error (-100001)');
      console.log(`reason: ${error.message ?? 'unknown'}`);
      // data might contains recent context (add to log just in case))
      if (data) {
        console.log('[UITarsModel] last context (truncated):');
        console.log(JSON.stringify(data).slice(0, 1000));   // truncate to 1000 characters
      }
      return process.exit(101);  // return to Life-time Manager LLM
    }

    console.error('[UI-TARS] error', error, data);
    process.exit(1);
  },

  maxLoopCount     : life_time,
  loopIntervalInMs : 1000,       // Time interval between two loop iterations (in milliseconds)  default: 0 
  uiTarsVersion    : UITarsModelVersion.V1_5,   // I hardcoded. Task: import converter like getModelVersion(not exported though)
});


(async () => {
  try {
    await agent.run(paramJSON.task);
    process.exit(0);       // successfully exit. example: Action finished()
  } catch (err: any) {
    /* error which is not handled in onError */
    console.error('[UI-TARS] fatal', err);
    process.exit(1);
  }
})();
